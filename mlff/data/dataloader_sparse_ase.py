from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import iread
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import jraph

import logging

from functools import partial, partialmethod

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


@dataclass
class AseDataLoaderSparse:
    input_file: str

    def cardinality(self):
        n = 0
        for _ in iread(self.input_file):
            n += 1
        return n

    def load(self, cutoff: float, pick_idx: np.ndarray = None):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        logging.mlff(
            f"Load data from {self.input_file} and calculate neighbors within cutoff={cutoff} Ang ..."
        )
        loaded_data = []
        max_num_of_nodes = 0
        max_num_of_edges = 0
        i = 0
        for a in tqdm(iread(self.input_file)):
            if keep(i):
                graph = ASE_to_jraph(a, cutoff=cutoff)
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


def ASE_to_jraph(
    mol: Atoms,
    cutoff: float,
    self_interaction: bool = False,
) -> jraph.GraphsTuple:
    """Convert an ASE Atoms object to a jraph.GraphTuple object.

    Args:
        mol (Atoms): ASE Atoms object.
        cutoff (float): Cutoff radius for neighbor interactions.
        self_interaction (bool): Include self-interaction in neighbor list.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object.
    """

    atomic_numbers = mol.get_atomic_numbers()
    positions = mol.get_positions()
    if mol.get_calculator() is not None:
        try:
            energy = mol.get_potential_energy()
        except PropertyNotImplementedError:
            energy = None
        try:
            forces = mol.get_forces()
        except PropertyNotImplementedError:
            forces = None
        try:
            stress = mol.get_stress()
        except PropertyNotImplementedError:
            stress = None
        try:
            dipole_moment = mol.get_dipole_moment()
        except PropertyNotImplementedError:
            stress = None
    else:
        energy = None
        forces = None
        stress = None
        dipole = None

    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=self_interaction)
        cell = np.array(mol.get_cell())
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
            "atomic_numbers": np.array(atomic_numbers),
            "forces": np.array(forces) if forces is not None else None,
            }

    senders = np.array(j)
    receivers = np.array(i)

    n_node = np.array([mol.get_global_number_of_atoms()])
    n_edge = np.array([len(i)])

    global_context = {
        "energy": np.array([energy]) if energy is not None else None,
        "stress": np.array(stress) if stress is not None else None,
        "dipole_components": np.array(dipole) if dipole is not None else None
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
