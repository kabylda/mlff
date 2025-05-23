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
        max_num_theory_levels: int = 3, #get from the config
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
    if 'theory_level' in properties:
        theory_level = tf.reshape(element['theory_level'], (1,))
        globals_dict['theory_level'] = theory_level
        theory_mask = tf.one_hot(theory_level, depth=max_num_theory_levels)  # (1, num_theory_levels)
        globals_dict['theory_mask'] = theory_mask
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
    num_train: list
    num_valid: list
    mode: str
    input_folders: list
    dataset_weights: list

class StopToken:
    """Token to signal that a worker has finished processing."""
    pass

class QCMLDataLoaderSparseParallel:
    """Data loader for TFDS datasets that loads data in parallel."""

    STOP_TOKEN = StopToken()

    def __init__(self, config, input_folders, dataset_weights, length_unit, energy_unit):
        """Initialize the data loader with configuration parameters.
        
        Args:
            config: The configuration object containing all necessary parameters.
            input_folders: List of paths to the input data folders.
            dataset_weights: List of weights for each dataset.
            length_unit: Unit for length measurements.
            energy_unit: Unit for energy measurements.
        """
        self.config = config
        self.input_folders = input_folders
        self.dataset_weights = dataset_weights
        self.length_unit = length_unit
        self.energy_unit = energy_unit

        # Set parameters from config
        self.calculate_neighbors_lr = config.data.neighbors_lr_bool
        self.cutoff = config.model.cutoff / self.length_unit
        self.cutoff_lr = config.data.neighbors_lr_cutoff / self.length_unit if self.calculate_neighbors_lr else None
        self.max_force_filter = config.data.filter.max_force / self.energy_unit * self.length_unit if hasattr(config.data.filter, 'max_force') else 1.e6
        self.train_seed = config.training.training_seed
        self.batch_max_num_nodes = config.training.batch_max_num_nodes
        self.batch_max_num_edges = config.training.batch_max_num_edges
        self.batch_max_num_graphs = config.training.batch_max_num_graphs
        self.batch_max_num_pairs = config.training.batch_max_num_pairs if config.training.batch_max_num_pairs is not None else 0

        # Get per-dataset train/valid sizes
        if hasattr(config.data, 'datasets'):
            self.num_train = [d.get('num_train', 0) for d in config.data.datasets]
            self.num_valid = [d.get('num_valid', 0) for d in config.data.datasets]
            print(f"Total num_train: {sum(self.num_train)}")
            print(f"Total num_valid: {sum(self.num_valid)}")
        else:
            self.num_train = [config.training.num_train]
            self.num_valid = [config.training.num_valid]
            print(f"Total num_train: {sum(self.num_train)}")
            print(f"Total num_valid: {sum(self.num_valid)}")

        #TODO: add checks for num_train and num_valid
        try:
            self.n_proc = int(config.training.batch_n_proc)
            print(f"Using {self.n_proc} processes for parallel data loading")
        except:
            self.n_proc = 8
            print("Warning: No number of processes specified. Defaulting to 8.")

        # multithread stuff # important
        ctx = get_context("spawn")
        self.manager = ctx.Manager()
        self.executor = ProcessPoolExecutor(max_workers=self.n_proc, mp_context=ctx)

        # Cache for cardinality
        self._cardinality = None

    def cardinality(self) -> int:
        """Calculate total number of examples across all datasets.
        
        Returns:
            Total number of examples in all datasets combined.
        """
        if self._cardinality is not None:
            return self._cardinality

        total_examples = 0
        for i, folder in enumerate(self.input_folders):
            builder = tfds.builder_from_directory(folder)
            
            # Calculate train examples
            split = f'train'
            dataset = builder.as_dataset(split=split)
            count = tf.data.experimental.cardinality(dataset).numpy()
            
            total_examples += count
        
        self._cardinality = total_examples
        return total_examples

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
            train_seed: Random seed for shuffling
            
        Returns:
            Batched dataset of graph tuples
        """
        # First create graph tuples
        dataset = dataset.map(
            lambda element: create_graph_tuple_tf(
                element,
                cutoff=cutoff,
                calculate_neighbors_lr=calculate_neighbors_lr,
                cutoff_lr=cutoff_lr,
                max_num_theory_levels=16
            ), 
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Shuffle
        dataset = dataset.shuffle(
            buffer_size=10_000,
            reshuffle_each_iteration=True,
            seed=train_seed
        )

        # Prefetch for better performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Apply force filter
        dataset = dataset.filter(
            lambda graph: tf.math.less(tf.math.reduce_max(graph.nodes['forces']), tf.constant(max_force_filter))
        )
        #TODO: need to add data.transformations.unit_conversion_graph before filtering 

        # Create batches
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

            # Load all datasets
            datasets = []
            for i, folder in enumerate(config.input_folders):
                builder = tfds.builder_from_directory(folder)
                # Use per-dataset splits
                if config.mode == 'train':
                    split = f'train[:{config.num_train[i]}]'
                else:  # validation
                    split = f'train[-{config.num_valid[i]}:]'
                dataset = builder.as_dataset(split=split, shuffle_files=True, read_config=read_config)

                if config.mode == 'train':
                    dataset = dataset.repeat() # to avoid exhausting the smaller dataset, makes one epoch infinite

                datasets.append(dataset)

            # Combine datasets with weighted sampling
            dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=config.dataset_weights)

            # Shard the combined dataset
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
                # # Get number of non-padded graphs and theory levels
                # num_graphs = batch['num_of_non_padded_graphs']
                # theory_levels = batch['theory_level']
                # theory_mask = batch['theory_mask']
                # graph_mask = batch['graph_mask']
                
                # # Count graphs per theory level
                # # theory_counts = np.sum(theory_mask, axis=0)
                # total_graphs = jnp.sum(graph_mask)
                # graphs_per_level = jnp.sum(theory_mask * graph_mask[:, None], axis=0)
                # # print(f"Batch contains {num_graphs} graphs")
                # # for i, count in enumerate(theory_counts):
                # #     print(f"  - Theory level {i}: {count} graphs")
                # if wandb.run is not None:
                #     wandb.log({"total_graphs": total_graphs})
                #     wandb.log({"graphs_per_level_0": graphs_per_level[0]})
                #     wandb.log({"graphs_per_level_1": graphs_per_level[1]})
                #     wandb.log({"graphs_per_level_2": graphs_per_level[2]})
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
            mode: 'train' or 'validation'
            
        Yields:
            Batches of data
        """
        shuffle_seed = np.random.randint(0, 2**31 - 1) # different shuffle seed for each epoch
        output_queue = self.manager.Queue(maxsize=4) # cache maximum of 4 batches

        for i in range(self.n_proc):
            config = WorkerConfig(
                worker_idx=i,
                n_workers=self.n_proc,
                split=split,
                shuffle_seed=shuffle_seed,
                batch_max_num_nodes=self.batch_max_num_nodes,
                batch_max_num_edges=self.batch_max_num_edges,
                batch_max_num_graphs=self.batch_max_num_graphs,
                batch_max_num_pairs=self.batch_max_num_pairs,
                cutoff=self.cutoff,
                calculate_neighbors_lr=self.calculate_neighbors_lr,
                cutoff_lr=self.cutoff_lr,
                max_force_filter=self.max_force_filter,
                train_seed=self.train_seed,
                input_folders=self.input_folders,
                dataset_weights=self.dataset_weights,
                num_train=self.num_train,
                num_valid=self.num_valid,
                mode=mode
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

            if ctr % 1000 == 0:
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
        else:
            return self._generator_sync(split, mode)

    def _generator_sync(self, split, mode):
        """Synchronous data generator for single-process operation.
        
        Args:
            split: Dataset split to use
            
        Yields:
            Batches of data
        """
        # Load all datasets
        datasets = []
        for i, folder in enumerate(self.input_folders):
            builder = tfds.builder_from_directory(folder)
            # Use per-dataset splits
            if mode == 'train':
                split = f'train[:{self.num_train[i]}]'
            else:  # validation
                split = f'train[-{self.num_valid[i]}:]'
            dataset = builder.as_dataset(split=split, shuffle_files=True)
            datasets.append(dataset)

        # Combine datasets with weighted sampling
        dataset = tf.data.Dataset.sample_from_datasets(datasets) #, weights=self.dataset_weights)

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
