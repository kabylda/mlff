import jraph
import logging
import numpy as np

from dataclasses import dataclass
from functools import partial, partialmethod
from typing import Optional
import queue
import wandb
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, get_context

try:
    import tensorflow as tf
except ModuleNotFoundError:
    logging.warning(
        "For using QCMLDataLoader please install tensorflow."
    )
try:
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    logging.warning(
        "For using QCMLDataLoader please install tensorflow_datasets."
    )


logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


def compute_edges_tf(
    positions,
    cutoff: float
):
    """Compute edges between atoms based on distance cutoff.
    
    Args:
        positions: Atom positions tensor
        cutoff: Distance cutoff for edge creation
        
    Returns:
        centers: Indices of center atoms
        others: Indices of neighbor atoms
    """
    num_atoms = tf.shape(positions)[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = tf.norm(displacements, axis=-1)
    mask = ~tf.eye(num_atoms, dtype=tf.bool)  # Get rid of self-connections.
    keep_edges = tf.where((distances < cutoff) & mask)
    centers = tf.cast(keep_edges[:, 0], dtype=tf.int32)  # center indices
    others = tf.cast(keep_edges[:, 1], dtype=tf.int32)  # neighbor indices
    return centers, others


def create_graph_tuple_tf(
        element,
        cutoff: float,
        calculate_neighbors_lr: bool = False,
        cutoff_lr: Optional[float] = None,
) -> jraph.GraphsTuple:

    """Takes a data element and wraps relevant components in a GraphsTuple."""
    nodes_dict = dict()
    globals_dict = dict()

    atomic_numbers = element['atomic_numbers']
    positions = element['positions']

    nodes_dict['positions'] = positions
    nodes_dict['atomic_numbers'] = atomic_numbers

    properties = element.keys()

    if 'energy' in properties:
        globals_dict['energy'] = tf.reshape(element['energy'], (1,))
    if 'forces' in properties:
        nodes_dict['forces'] = element['forces']
    if 'hirshfeld_ratios' in properties:
        nodes_dict['hirshfeld_ratios'] = element['hirshfeld_ratios']
    else:
        # Hack to deal with symbolic tensors, since this depends on the number of atoms in the molecule.
        hirshfeld_ratios = tf.reshape(
            tf.zeros_like(atomic_numbers, dtype=tf.float32) * tf.constant([np.nan], dtype=tf.float32),
            (-1, )
        )
        globals_dict['hirshfeld_ratios'] = hirshfeld_ratios
    if 'multiplicity' in properties:
        globals_dict['num_unpaired_electrons'] = tf.reshape(element['multiplicity'], (1,)) - 1
    if 'charge' in properties:
        globals_dict['total_charge'] = tf.reshape(element['charge'], (1,))
    if 'stress' in properties:
        globals_dict['stress'] = tf.reshape(element['stress'], (1, 6))
    else:
        stress = np.empty((1, 6))
        stress[:] = np.nan
        globals_dict['stress'] = tf.convert_to_tensor(stress, dtype=tf.float32)
    if 'dipole_vec' in properties:
        globals_dict['dipole_vec'] = tf.reshape(element['dipole_vec'], (1, 3))
    else:
        dipole_vec = np.empty((1, 3))
        dipole_vec[:] = np.nan
        globals_dict['dipole_vec'] = tf.convert_to_tensor(dipole_vec, dtype=tf.float32)

    centers, others = compute_edges_tf(
        positions=positions,
        cutoff=cutoff
    )

    if calculate_neighbors_lr is True:
        if cutoff_lr is None:
            raise ValueError(
                f'cutoff_lr must be specified for {calculate_neighbors_lr=}. Received {cutoff_lr=}.'
            )
        centers_lr, others_lr = compute_edges_tf(
            positions=positions,
            cutoff=cutoff_lr
        )
    else:
        centers_lr = tf.constant([], dtype=tf.int64)
        others_lr = tf.constant([], dtype=tf.int64)

    num_edges_lr = tf.shape(centers_lr)[0]
    num_nodes = tf.shape(atomic_numbers)[0]
    num_edges = tf.shape(centers)[0]

    return jraph.GraphsTuple(
        n_node=tf.reshape(num_nodes, (1,)),
        n_edge=tf.reshape(num_edges, (1,)),
        # Central nodes (idx_i) receive information from the neighboring nodes (idx_j).
        receivers=centers,
        senders=others,
        nodes=nodes_dict,
        globals=globals_dict,
        edges=dict(),  # Don't set to None, since otherwise tf.data.Dataset.to_numpy_generator() does not work due to
        # call to None.numpy().
        idx_i_lr=centers_lr,
        idx_j_lr=others_lr,
        n_pairs=tf.reshape(num_edges_lr, (1,))
    )
    

@dataclass
class WorkerConfig:
    """Configuration for worker processes.
    
    Contains all parameters needed by worker processes to load and process data.
    """
    input_folder: str
    split: str
    cutoff: float
    batch_max_num_nodes: int
    batch_max_num_edges: int
    batch_max_num_graphs: int
    batch_max_num_pairs: int
    shuffle_seed: int
    n_workers: int
    worker_idx: int
    max_force_filter: float
    calculate_neighbors_lr: bool
    cutoff_lr: float
    train_seed: int
    
class StopToken:
    """Token to signal that a worker has finished processing."""
    pass

class QCMLDataLoaderSparseParallel:
    """Parallel data loader for the QCML dataset.
    
    Uses multiple processes to load and preprocess data in parallel.
    """
    STOP_TOKEN = StopToken()

    def __init__(self, **kwargs):
        """Initialize the data loader with configuration parameters."""
        self.data_cfg = kwargs  # Store config in data_cfg attribute

        self.input_folder = self.data_cfg["input_folder"]

        # check dataset version
        builder = tfds.builder_from_directory(self.input_folder)
 
        self.cutoff = self.data_cfg["cutoff"]
        self.batch_max_num_nodes = self.data_cfg["batch_max_num_nodes"]
        self.batch_max_num_edges = self.data_cfg["batch_max_num_edges"]
        self.batch_max_num_graphs = self.data_cfg["batch_max_num_graphs"]
        self.batch_max_num_pairs = self.data_cfg["batch_max_num_pairs"]
        self.max_force_filter = self.data_cfg["max_force_filter"]
        self.calculate_neighbors_lr = self.data_cfg["calculate_neighbors_lr"]
        self.cutoff_lr = self.data_cfg["cutoff_lr"]
        self.train_seed = self.data_cfg["train_seed"]

        if "n_proc" in self.data_cfg:
            try:
                self.n_proc = int(self.data_cfg["n_proc"][0])  # Access first element
            except:
                self.n_proc = int(self.data_cfg["n_proc"])  # Access first element

        else:
            print("Warning: No number of processes specified. Defaulting to 8.")
            self.n_proc = 8

        # multithread stuff # important
        ctx = get_context("spawn")
        self.manager = ctx.Manager()
        self.executor = ProcessPoolExecutor(max_workers=self.n_proc, mp_context=ctx)

    @staticmethod
    def _preprocess(dataset, 
                    batch_max_num_nodes, 
                    batch_max_num_edges, 
                    batch_max_num_graphs, 
                    batch_max_num_pairs,
                    cutoff,
                    calculate_neighbors_lr,
                    cutoff_lr,
                    max_force_filter,
                    train_seed):
        """Preprocess the dataset by creating graph tuples and batching.
        
        Args:
            dataset: TensorFlow dataset to preprocess
            batch_max_num_nodes: Maximum number of nodes in a batch
            batch_max_num_edges: Maximum number of edges in a batch
            batch_max_num_graphs: Maximum number of graphs in a batch
            batch_max_num_pairs: Maximum number of pairs in a batch
            cutoff: Distance cutoff for edge creation
            calculate_neighbors_lr: Whether to calculate long-range neighbors
            cutoff_lr: Distance cutoff for long-range neighbors
            max_force_filter: Maximum force value for filtering
            
        Returns:
            Batched dataset of graph tuples
        """
        dataset = dataset.map(
            lambda element: create_graph_tuple_tf(
                element,
                cutoff=cutoff,
                calculate_neighbors_lr=calculate_neighbors_lr,
                cutoff_lr=cutoff_lr
            ), 
            num_parallel_calls=tf.data.AUTOTUNE,
        ).shuffle(
            buffer_size=10_000,
            reshuffle_each_iteration=True,
            seed=train_seed 
        ).prefetch(tf.data.AUTOTUNE
        ).filter(
            lambda graph: tf.math.less(tf.math.reduce_max(graph.nodes['forces']), tf.constant(max_force_filter))
        )
        #TODO: need to add data.transformations.unit_conversion_graph before filtering 

        batched_dataset = jraph.dynamically_batch(
            dataset.as_numpy_iterator(),
            n_node=batch_max_num_nodes,
            n_edge=batch_max_num_edges,
            n_graph=batch_max_num_graphs,
            n_pairs=batch_max_num_pairs
        )

        return batched_dataset

    @staticmethod
    def _safe_put(queue, item):
        """Safely put an item in a queue, handling potential errors.
        
        Args:
            queue: Queue to put the item into
            item: Item to put in the queue
        """
        try:
            queue.put(item)
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            print(f"[!] Queue closed before item could be put: {e}")
        except Exception as e:
            print(f"[!] Unexpected error putting item to queue: {e}")

    @staticmethod
    def _worker(config, output_queue):
        """Worker function that loads data for the given indices and puts it into the queue.
        
        Args:
            config: WorkerConfig with all parameters
            output_queue: Queue to put processed batches into
        """
        try:
            # We need a deterministic shuffle seed, s.t. workers shuffle files in the same way
            read_config = tfds.ReadConfig(
                shuffle_seed=config.shuffle_seed,
            )

            # Note that tfds automatically pre-fetches after reading
            # This might be suboptimal if we prefetch later and we can try to disable it
            builder = tfds.builder_from_directory(config.input_folder)
            dataset = builder.as_dataset(split=config.split, shuffle_files=True, read_config=read_config)

            dataset = dataset.shard(num_shards=config.n_workers, index=config.worker_idx)

            for batch in QCMLDataLoaderSparseParallel._preprocess(dataset,   
                                                batch_max_num_nodes=config.batch_max_num_nodes,
                                                batch_max_num_edges=config.batch_max_num_edges,
                                                batch_max_num_graphs=config.batch_max_num_graphs,
                                                batch_max_num_pairs=config.batch_max_num_pairs,
                                                cutoff=config.cutoff,
                                                calculate_neighbors_lr=config.calculate_neighbors_lr,
                                                cutoff_lr=config.cutoff_lr,
                                                max_force_filter=config.max_force_filter,
                                                train_seed=config.train_seed
                                                ):
                output_queue.put(batch)
        except Exception as e:
            print(f"[!] Error in worker {config.worker_idx}: {e}")
        finally:
            # Finished processing data, always put a stop token
            QCMLDataLoaderSparseParallel._safe_put(output_queue, QCMLDataLoaderSparseParallel.STOP_TOKEN)
    
    def _generator(self, split, mode):
        """Generator reads data from the queue.
        
        Args:
            split: Dataset split to use
            
        Yields:
            Batches of data
        """
        shuffle_seed = np.random.randint(0, 2**31 - 1) # different shuffle seed for each epoch
        output_queue = self.manager.Queue(maxsize=4) # cache maximum of 4 batches

        for i in range(self.n_proc):
            config = WorkerConfig(
                input_folder=self.input_folder,
                split=split,
                cutoff=self.cutoff,
                batch_max_num_nodes=self.batch_max_num_nodes,
                batch_max_num_edges=self.batch_max_num_edges,
                batch_max_num_graphs=self.batch_max_num_graphs,
                batch_max_num_pairs=self.batch_max_num_pairs,
                shuffle_seed=shuffle_seed,
                n_workers=self.n_proc,
                worker_idx=i,
                max_force_filter=self.max_force_filter,
                train_seed=self.train_seed,
                calculate_neighbors_lr=self.calculate_neighbors_lr,
                cutoff_lr=self.cutoff_lr
            )
            self.executor.submit(QCMLDataLoaderSparseParallel._worker, config, output_queue)
    
        n_stop_token = 0
        ctr = 0
        while True:
            ctr += 1

            try:
                batch = output_queue.get(timeout=3)
            except queue.Empty:
                # Timeout expired, try again
                continue
            except Exception as e:
                print(f"[!] Unexpected error retrieving item from queue: {e}")
                break

            if ctr % 100 == 0:
                size = output_queue.qsize()
                if wandb.run is not None:
                    wandb.log({"queue_size_"+mode: size})

            if isinstance(batch, StopToken):
                n_stop_token += 1

                # wait for all processes to finish
                if n_stop_token == self.n_proc:
                    break
            else:
                yield batch

    def next_epoch(self, split, mode):
        """Loads the data for ONE epoch.
        
        Args:
            split: Dataset split to use
            
        Returns:
            Generator yielding batches of data
        """
        if mode == 'train': # use multiporcessing for training batch 
            return self._generator(split, mode)
        else: #for now use single process for validation/test batch
            return self._generator_sync(split)

    def _generator_sync(self, split):
        """Synchronous data generator for single-process operation.
        
        Args:
            split: Dataset split to use
            
        Yields:
            Batches of data
        """
        builder = tfds.builder_from_directory(self.input_folder)
        dataset = builder.as_dataset(split=split, shuffle_files=True)

        for batch in self._preprocess(dataset, 
                                     batch_max_num_nodes=self.batch_max_num_nodes,
                                     batch_max_num_edges=self.batch_max_num_edges,
                                     batch_max_num_graphs=self.batch_max_num_graphs,
                                     batch_max_num_pairs=self.batch_max_num_pairs,
                                     cutoff=self.cutoff,
                                     calculate_neighbors_lr=self.calculate_neighbors_lr,
                                     cutoff_lr=self.cutoff_lr,
                                     max_force_filter=self.max_force_filter,
                                     train_seed=self.train_seed
                                    ):
            yield batch
