from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
import jraph
import numpy as np
from tqdm import tqdm

import logging

from functools import partial, partialmethod

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


@dataclass
class NpzDataLoaderSparse:
    input_file: str

    def cardinality(self):
        return len(np.load(self.input_file)['positions'])

    def load(self, cutoff: float, pick_idx: np.ndarray = None):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        np_data = np.load(self.input_file)

        keys = list(np_data.keys())
        data_iterator = zip(*(np_data[k] for k in keys))

        loaded_data = []
        max_num_of_nodes = 0
        max_num_of_edges = 0
        logging.mlff(
            f"Load data from {self.input_file} and calculate neighbors within cutoff={cutoff} Ang ..."
        )
        i = 0
        for tup in tqdm(data_iterator):
            if keep(i):
                entry = {k: v for (k, v) in zip(keys, tup)}
                graph = entry_to_jraph(entry, cutoff=cutoff)
                num_nodes = len(graph.nodes['atomic_numbers'])
                num_edges = len(graph.receivers)
                max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges
                loaded_data.append(graph)
            else:
                pass
            i += 1

        if pick_idx is not None:
            if max(pick_idx) >= i:
                raise RuntimeError(
                    f'`max(pick_idx) = {max(pick_idx)} >= cardinality = {i} of the dataset`.'
                )
        logging.mlff("... done!")

        return loaded_data, {'max_num_of_nodes': max_num_of_nodes, 'max_num_of_edges': max_num_of_edges}


def entry_to_jraph(
    entry,
    cutoff: float,
    self_interaction: bool = False,
) -> jraph.GraphsTuple:
    """Convert an entry to a jraph.GraphTuple object.

    Args:
        entry (Dict): Entry.
        cutoff (float): Cutoff radius for neighbor interactions.
        self_interaction (bool): Include self-interaction in neighbor list.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object.
    """

    atomic_numbers = entry['atomic_numbers']
    positions = entry['positions']
    total_charge = entry.get('total_charge')
    num_unpaired_electrons = entry.get('num_unpaired_electrons')
    forces = entry.get('forces')
    energy = entry.get('energy')
    stress = entry.get('stress')
    cell = entry.get('cell')
    pbc = entry.get('pbc')
    mol = Atoms(positions=positions, numbers=atomic_numbers, cell=cell, pbc=pbc)

    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=self_interaction)
        edge_features = {
            "cell": np.repeat(np.array(cell)[None], repeats=len(S), axis=0),
            "cell_offset": np.array(S)
        }
    else:
        i, j = neighbor_list('ij', mol, cutoff, self_interaction=self_interaction)
        edge_features = {
            "cell": None,
            "cell_offset": None
        }

    node_features = {
        "positions": np.array(positions),
        "atomic_numbers": np.array(atomic_numbers, dtype=np.int64),
        "forces": np.array(forces),
    }

    senders = np.array(j)
    receivers = np.array(i)

    n_node = np.array([mol.get_global_number_of_atoms()])
    n_edge = np.array([len(i)])

    global_context = {
        "energy": np.array(energy).reshape(-1) if energy is not None else None,
        "stress": np.array(stress) if stress is not None else None,
        "total_charge": np.array(total_charge, dtype=np.int16).reshape(-1) if total_charge is not None else None,
        "num_unpaired_electrons": np.array(num_unpaired_electrons, dtype=np.int16).reshape(-1) if num_unpaired_electrons is not None else None
    }

    return jraph.GraphsTuple(
                nodes=node_features,
                edges=edge_features,
                senders=senders,
                receivers=receivers,
                n_node=n_node,
                n_edge=n_edge,
                globals=global_context
    )
