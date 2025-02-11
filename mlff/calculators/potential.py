"""Potential calculator."""
import jax
import jax.numpy as jnp
import logging
import numpy as np

from abc import abstractmethod
from flax import struct
from typing import Any, Callable, Type, Dict

from glp.graph import Graph

from ..utils import calculator_utils
from .base_calculator import make_base_calculator


@struct.dataclass
class MachineLearningPotential:
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)

    @classmethod
    @abstractmethod
    def create_from_workdir(cls, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, inputs: Dict):
        pass


@struct.dataclass
class PotentialSparse(MachineLearningPotential):
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    long_range_bool: bool = struct.field(pytree_node=False)
    long_range_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph, bool], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)


    @classmethod
    def create_from_workdir(
            cls,
            workdir: str,
            from_file: bool = False,
            add_shift: bool = False,
            long_range_kwargs: Dict[str, Any] = None,
            dtype=jnp.float32,
            model: str = 'so3krates',
    ):
        """


        Args:
            workdir ():
            from_file ():
            add_shift ():
            long_range_kwargs (): Dictionary with keyword arguments for the long-range modules.
            dtype ():
            model ():

        Returns:

        """

        if add_shift is True and (dtype == np.float32 or dtype == jnp.float32):
            logging.warning(
                'Energy shift is enabled but float32 precision is used.'
                ' For large absolute energy values, this can lead to floating point errors in the energies.'
                ' If you do not need the absolute energy values since only relative ones are important, we'
                ' suggest to disable the energy shift since increasing the precision slows down'
                ' computation.'
            )

        cfg = calculator_utils.load_hyperparameters(workdir=workdir)

        long_range_bool = (cfg.model.electrostatic_energy_bool is True) or (cfg.model.dispersion_energy_bool is True)

        cutoff = cfg.model.cutoff

        # ITPNet is strictly local so has effectively "one" MP layer in terms of effective cutoff.
        steps = cfg.model.num_layers if model != 'itp_net' else 1

        effective_cutoff = steps * cutoff

        if add_shift:
            shifts = {int(k): float(v) for k, v in dict(cfg.data.energy_shifts).items()}

            def shift(v, z):
                return v + jnp.asarray(shifts, dtype=dtype)[z][:, None]
        else:
            def shift(v, z):
                return v

        def shift_fn(x: jnp.ndarray, z: jnp.ndarray):
            return shift(x, z)

        base_calculator_fn = make_base_calculator(
            workdir=workdir,
            model=model,
            long_range_kwargs=long_range_kwargs,
            calculate_forces=False,
            input_convention='displacements',
            output_convention='per_atom',
            from_file=from_file
        )

        def potential_fn(
                graph: Graph,
                has_aux: bool = False
        ):

            x = glp_graph_to_mlff_input(
                graph,
                long_range_bool=long_range_bool
            )

            y = base_calculator_fn(x)

            energy = y['energy']
            atomic_numbers = x['atomic_numbers']

            # Shift function either applies a shift or not, depending on `add_shifts`
            shifted_energy = shift_fn(energy, atomic_numbers).reshape(-1).astype(dtype)

            if has_aux:

                aux = jax.tree_map(lambda u: u.astype(dtype), y)
                aux.pop('energy')

                return (
                    shifted_energy,
                    aux
                )

            else:

                return shifted_energy

        return cls(
            cutoff=cutoff,
            effective_cutoff=effective_cutoff,
            long_range_bool=long_range_bool,
            long_range_cutoff=long_range_kwargs['cutoff_lr'] if long_range_bool is True else None,
            potential_fn=potential_fn,
            dtype=dtype
        )

    def __call__(
            self,
            graph: Graph,
            has_aux: bool = False
    ) -> jnp.ndarray:

        return self.potential_fn(
            graph,
            has_aux
        )


def glp_graph_to_mlff_input(graph: Graph, long_range_bool: bool):
    # Local case corresponds to no having idx_lr_i, idx_lr_j, displacements_lr = None, None, None
    x = {
        'positions': graph.positions,
        'displacements': graph.edges,
        'atomic_numbers': graph.nodes,
        'idx_i': graph.centers,
        'idx_j': graph.others,
        'total_charge': graph.total_charge,
        'num_unpaired_electrons': graph.num_unpaired_electrons,
        'cell': getattr(graph, 'cell', None),
    }
    if long_range_bool is True:
        x_lr = {
            'displacements_lr': graph.edges_lr,
            'idx_i_lr': graph.idx_i_lr,
            'idx_j_lr': graph.idx_j_lr,
        }

        x.update(x_lr)

    return x