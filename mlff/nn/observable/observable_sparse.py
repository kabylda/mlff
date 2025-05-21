import flax.linen as nn
import jax
import jax.numpy as jnp

from ase.units import Bohr, Hartree
from ase.units import alpha as fine_structure
from functools import partial
from jax.ops import segment_sum
from jax.nn.initializers import constant
from typing import Any, Callable, Dict, Tuple
from typing import Optional

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_mask
from mlff.nn.observable.dispersion_ref_data import alphas, C6_coef
from mlff.masking.mask import safe_scale
from mlff.nn.activation_function.activation_function import softplus_inverse, softplus


class EnergySparse(BaseSubModule):
    prop_keys: Dict
    zmax: int = 118
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    learn_atomic_type_scales: bool = False
    learn_atomic_type_shifts: bool = False
    output_is_zero_at_init: bool = True
    output_convention: str = 'per_structure'
    module_name: str = 'energy'
    electrostatic_energy_bool: bool = False
    electrostatic_energy: Optional[Any] = None
    dispersion_energy_bool: bool = False
    dispersion_energy: Optional[Any] = None
    zbl_repulsion_bool: bool = False
    zbl_repulsion: Optional[Any] = None
    use_final_bias_bool: bool = True

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        #TODO: Update docstring
        Args:
            inputs (Dict):
                x (Array): Node features, (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, (num_nodes)
                batch_segments (Array): Batch segments, (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                graph_mask (Array): Graph mask, (num_graphs)
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        theory_mask = inputs['theory_mask'] 
        num_theory_levels = theory_mask.shape[-1]

        num_graphs = len(graph_mask)
        if self.learn_atomic_type_shifts:
            energy_offset = jnp.take(
                self.param(
                    'energy_offset',
                    nn.initializers.zeros_init(),
                    (self.zmax + 1, num_theory_levels)
                ),
                atomic_numbers,
                axis=0
            )  # (num_nodes, num_levels_of_theory)
        else:
            energy_offset = jnp.zeros((1,), dtype=x.dtype)

        if self.learn_atomic_type_scales:
            atomic_scales = jnp.take(
                self.param(
                    'atomic_scales',
                    nn.initializers.ones_init(),
                    (self.zmax + 1, num_theory_levels)
                ), atomic_numbers, axis=0)  # (num_nodes, num_levels_of_theory)
        else:
            atomic_scales = jnp.ones((1,), dtype=x.dtype)

        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='energy_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            atomic_energy = nn.Dense(
                num_theory_levels,
                kernel_init=self.kernel_init,
                # optional
                use_bias=False,
                name='energy_dense_final'
            )(y)  # (num_nodes, num_levels_of_theory)
        else:
            atomic_energy = nn.Dense(
                num_theory_levels,
                # optional
                use_bias=False,
                kernel_init=self.kernel_init,
                name='energy_dense_final'
            )(x) # (num_nodes, num_levels_of_theory)

        atomic_energy = atomic_energy * atomic_scales
        atomic_energy += energy_offset  # (num_nodes, num_levels_of_theory)

        atomic_energy = jnp.where(
            theory_mask,
            atomic_energy,
            jnp.zeros_like(atomic_energy)
        ).sum(axis=-1)  # (num_nodes)

        atomic_energy = safe_scale(atomic_energy, node_mask)

        if self.zbl_repulsion_bool:
            repulsion_energy = self.zbl_repulsion(inputs)['zbl_repulsion']
            atomic_energy += repulsion_energy

        if self.electrostatic_energy_bool:
            electrostatic_energy = self.electrostatic_energy(inputs)['electrostatic_energy']
            atomic_energy += electrostatic_energy

        if self.dispersion_energy_bool:
            dispersion_energy = self.dispersion_energy(inputs)['dispersion_energy']
            atomic_energy += dispersion_energy

        if self.output_convention == 'per_structure':
            energy = segment_sum(
                atomic_energy,
                segment_ids=batch_segments,
                num_segments=num_graphs
            )  # (num_graphs)
            energy = safe_scale(energy, graph_mask)

            return dict(energy=energy)

        elif self.output_convention == 'per_atom':
            energy = atomic_energy  # (num_nodes)
            return dict(energy=energy)

        else:
            raise ValueError(
                f'{self.output_convention} is invalid argument for attribute `output_convention`.'
            )

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'zmax': self.zmax,
                                   'output_is_zero_at_init': self.output_is_zero_at_init,
                                   'output_convention': self.output_convention,
                                   'zbl_repulsion_bool': self.zbl_repulsion_bool,
                                   'electrostatic_energy_bool': self.electrostatic_energy_bool,
                                   'dispersion_energy_bool': self.dispersion_energy_bool,
                                   'prop_keys': self.prop_keys}
                }


class HirshfeldSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name = 'hirshfeld_ratios'

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        #TODO: Update docstring
        Predict Hirshfeld volumes from atom-wise features `x` and atomic types `z`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (num_nodes, num_features)
                atomic_numbers (Array): Atomic types, shape: (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                theory_mask (Array): Theory mask, (num_nodes, num_theory_levels)
            *args ():
            **kwargs ():

        Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld ratios

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        theory_mask = inputs['theory_mask']  # (num_nodes, num_theory_levels)
        num_theory_levels = theory_mask.shape[-1]

        # Element-dependent bias for each theory level
        v_shift = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1) # shape: (num_nodes)

        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='hirshfeld_ratios_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            v_pred = nn.Dense(
                num_theory_levels,
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(y)  # (num_nodes, num_theory_levels)
        else:
            v_pred = nn.Dense(
                num_theory_levels,
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(x)  # (num_nodes, num_theory_levels)
        # Apply theory mask first
        v_pred_masked = jnp.where(
            theory_mask,
            v_pred,
            jnp.zeros_like(v_pred)
        ).sum(axis=-1)
        
        # Add shift and take absolute value
        v_eff = v_shift + v_pred_masked  # shape: (num_nodes)
        hirshfeld_ratios = jnp.abs(v_eff) # (num_nodes)
        hirshfeld_ratios = safe_scale(hirshfeld_ratios, node_mask)

        return dict(hirshfeld_ratios=hirshfeld_ratios)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


class PartialChargesSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'partial_charges'

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:

        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        total_charge = inputs['total_charge']  # (num_graphs)

        num_graphs = len(graph_mask)
        num_nodes = len(node_mask)

        # q_ - element-dependent bias
        q_ = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)

        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='charge_dense_regression_vec'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final_vec'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final_vec'
            )(x).squeeze(axis=-1)  # (num_nodes)

        x_q = safe_scale(x_ + q_, node_mask)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts=True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule,
                                           total_repeat_length=num_nodes)  # shape: (num_nodes)

        return dict(partial_charges=partial_charges)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


class DipoleVecSparse(BaseSubModule):
    prop_keys: Dict
    partial_charges: Optional[Any] = None
    module_name: str = 'dipole_vec'

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:

        batch_segments = inputs['batch_segments']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        positions = inputs['positions']  # (num_nodes, 3)

        num_graphs = len(graph_mask)

        # Calculate partial charges
        partial_charges = self.partial_charges(inputs)['partial_charges']

        if positions is None:
            # TODO: do not calculate DipoleVecSparse if there is no positions
            mu_i = 1 * partial_charges[:, None]
        else:
            mu_i = positions * partial_charges[:, None]

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = safe_scale(dipole, graph_mask[:, None])

        return dict(dipole_vec=dipole_vec)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention


@jax.jit
def sigma(x):
    return safe_mask(x > 0, fn=lambda u: jnp.exp(-1. / u), operand=x, placeholder=0)


@jax.jit
def switching_fn(x, x_on, x_off):
    c = (x - x_on) / (x_off - x_on)
    return sigma(1 - c) / (sigma(1 - c) + sigma(c))


@partial(jax.jit, static_argnames=('neighborlist_format',))
def vdw_QDO_disp_damp(
        R,
        gamma,
        C6,
        alpha_ij,
        gamma_scale,
        neighborlist_format: str = 'sparse'
):
    # Determine the input dtype
    input_dtype = R.dtype

    #  Compute the vdW-QDO dispersion energy (in eV)
    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    C8 = 5 / gamma * C6
    C10 = 245 / 8 / gamma ** 2 * C6
    p = gamma_scale * 2 * 2.54 * alpha_ij ** (1 / 7)

    C8 = jnp.asarray(C8, dtype=input_dtype)
    C10 = jnp.asarray(C10, dtype=input_dtype)
    p = jnp.asarray(p, dtype=input_dtype)

    V3 = -C6 / (jnp.power(R, 6) + jnp.power(p, 6)) - C8 / (jnp.power(R, 8) + jnp.power(p, 8)) - C10 / (
                jnp.power(R, 10) + jnp.power(p, 10))

    return c * V3 * jnp.asarray(Hartree, dtype=input_dtype)


@jax.jit
def mixing_rules(
        atomic_numbers: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        hirshfeld_ratios: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dtype = hirshfeld_ratios.dtype

    atomic_number_i = atomic_numbers[idx_i] - 1
    atomic_number_j = atomic_numbers[idx_j] - 1
    hirshfeld_ratio_i = hirshfeld_ratios[idx_i]
    hirshfeld_ratio_j = hirshfeld_ratios[idx_j]

    alpha_i = jnp.asarray(jnp.take(alphas, atomic_number_i, axis=0), dtype=dtype) * hirshfeld_ratio_i
    C6_i = jnp.asarray(jnp.take(C6_coef, atomic_number_i, axis=0), dtype=dtype) * jnp.square(hirshfeld_ratio_i)
    alpha_j = jnp.asarray(jnp.take(alphas, atomic_number_j, axis=0), dtype=dtype) * hirshfeld_ratio_j
    C6_j = jnp.asarray(jnp.take(C6_coef, atomic_number_j, axis=0), dtype=dtype) * jnp.square(hirshfeld_ratio_j)

    alpha_ij = (alpha_i + alpha_j) / 2
    C6_ij = 2 * C6_i * C6_j * alpha_j * alpha_i / (alpha_i ** 2 * C6_j + alpha_j ** 2 * C6_i)

    return alpha_ij, C6_ij


@jax.jit
def gamma_cubic_fit(alpha):
    input_dtype = alpha.dtype

    vdW_radius = fine_structure ** (jnp.asarray(-4. / 21, input_dtype)) * alpha ** jnp.asarray(1. / 7, input_dtype)
    b0 = jnp.asarray(-0.00433008, dtype=input_dtype)
    b1 = jnp.asarray(0.24428889, dtype=input_dtype)
    b2 = jnp.asarray(0.04125273, dtype=input_dtype)
    b3 = jnp.asarray(-0.00078893, dtype=input_dtype)

    sigma = b3 * jnp.power(vdW_radius, 3) + b2 * jnp.square(vdW_radius) + b1 * vdW_radius + b0
    gamma = jnp.asarray(1. / 2, dtype=input_dtype) / jnp.square(sigma)
    return gamma


@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_erf(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        sigma: float,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """ Pairwise Coulomb interaction with erf damping """
    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    # Cast constants to input dtype
    _ke = jnp.asarray(ke, dtype=input_dtype)
    _sigma = jnp.asarray(sigma, dtype=input_dtype)

    pairwise = c * _ke * q[idx_i] * q[idx_j] * jax.lax.erf(rij / _sigma) / rij

    return pairwise


@partial(jax.jit, static_argnames=('neighborlist_format',))
def coulomb_erf_shifted_force_smooth(
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        ke: float,
        sigma: float,
        cutoff: float,
        cuton: float,
        neighborlist_format: str = 'sparse'
) -> jnp.ndarray:
    """ Pairwise Coulomb interaction with erf damping, using Shifted Force method """

    input_dtype = rij.dtype

    if neighborlist_format == 'sparse':
        c = jnp.asarray(0.5, dtype=input_dtype)
    elif neighborlist_format == 'ordered_sparse':
        c = jnp.asarray(1.0, dtype=input_dtype)
    else:
        raise ValueError(
            f"neighborlist_format must be one of either 'ordered_sparse' or 'sparse'. "
            f"received {neighborlist_format=}"
        )

    # Cast the constants to input dtype
    _sigma = jnp.asarray(sigma, dtype=input_dtype)
    _ke = jnp.asarray(ke, dtype=input_dtype)
    _cutoff = jnp.asarray(cutoff, dtype=input_dtype)
    _cuton = jnp.asarray(cuton, dtype=input_dtype)

    def potential(r):
        return jax.lax.erf(r / _sigma) / r

    def force(r):
        return (2 * r * jnp.exp(-(r / _sigma) ** 2) / (jnp.sqrt(jnp.pi) * _sigma) - jax.lax.erf(r / _sigma)) / r ** 2

    f = switching_fn(rij, _cuton, _cutoff)
    pairwise = potential(rij)
    shift = potential(_cutoff)
    force_shift = force(_cutoff)

    shifted_potential = pairwise - shift - force_shift * (rij - _cutoff)

    return jnp.where(
        rij < _cutoff,
        c * _ke * q[idx_i] * q[idx_j] * (f * (pairwise - shift) + (1 - f) * shifted_potential),
        0.0
    )


class ZBLRepulsionSparse(BaseSubModule):
    """
    Ziegler-Biersack-Littmark repulsion.
    """
    prop_keys: Dict
    # input_convention: str = 'positions'
    module_name: str = 'zbl_repulsion'
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        a1 = softplus(self.param('a1', constant(softplus_inverse(3.20000)), (1,)))  # shape: (1)
        a2 = softplus(self.param('a2', constant(softplus_inverse(0.94230)), (1,)))  # shape: (1)
        a3 = softplus(self.param('a3', constant(softplus_inverse(0.40280)), (1,)))  # shape: (1)
        a4 = softplus(self.param('a4', constant(softplus_inverse(0.20160)), (1,)))  # shape: (1)
        c1 = softplus(self.param('c1', constant(softplus_inverse(0.18180)), (1,)))  # shape: (1)
        c2 = softplus(self.param('c2', constant(softplus_inverse(0.50990)), (1,)))  # shape: (1)
        c3 = softplus(self.param('c3', constant(softplus_inverse(0.28020)), (1,)))  # shape: (1)
        c4 = softplus(self.param('c4', constant(softplus_inverse(0.02817)), (1,)))  # shape: (1)
        p = softplus(self.param('p', constant(softplus_inverse(0.23)), (1,)))  # shape: (1)
        d = softplus(self.param('d', constant(softplus_inverse(1 / (0.8854 * self.a0))), (1,)))  # shape: (1)

        c_sum = c1 + c2 + c3 + c4
        c1 = c1 / c_sum
        c2 = c2 / c_sum
        c3 = c3 / c_sum
        c4 = c4 / c_sum

        atomic_numbers = inputs['atomic_numbers']
        node_mask = inputs['node_mask']
        phi_r_cut_ij = inputs['cut']
        idx_i = inputs['idx_i']
        idx_j = inputs['idx_j']
        d_ij = inputs['d_ij']

        num_nodes = len(node_mask)

        z_i = atomic_numbers[idx_i]
        z_j = atomic_numbers[idx_j]

        z_d_ij = safe_mask(mask=d_ij != 0,
                           operand=d_ij,
                           fn=lambda u: z_i * z_j / u,
                           placeholder=0.
                           )

        x = self.ke * phi_r_cut_ij * z_d_ij

        rzd = d_ij * (jnp.power(z_i, p) + jnp.power(z_j, p)) * d
        y = c1 * jnp.exp(-a1 * rzd) + c2 * jnp.exp(-a2 * rzd) + c3 * jnp.exp(-a3 * rzd) + c4 * jnp.exp(-a4 * rzd)

        w = switching_fn(d_ij, x_on=0, x_off=1.5)

        e_rep_edge = w * x * y / jnp.asarray(2, dtype=d_ij.dtype)
        e_rep_edge = segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=num_nodes)
        e_rep_edge = safe_scale(e_rep_edge, node_mask)

        return dict(zbl_repulsion=e_rep_edge)

    def reset_output_convention(self, output_convention):
        pass


class ElectrostaticEnergySparse(BaseSubModule):
    prop_keys: Dict
    partial_charges: Any
    cutoff_lr: float
    ke: float = 14.399645351950548
    electrostatic_energy_scale: float = 1.0
    neighborlist_format: str = 'sparse'  # or 'ordered_sparse'
    module_name: str = 'electrostatic_energy'

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        node_mask = inputs['node_mask']  # (num_nodes)
        num_nodes = len(node_mask)
        idx_i_lr = inputs['idx_i_lr']
        idx_j_lr = inputs['idx_j_lr']
        d_ij_lr = inputs['d_ij_lr']

        # Calculate partial charges
        partial_charges = self.partial_charges(inputs)['partial_charges']

        # If cutoff is set, we apply damping with error function with smoothing to zero at cutoff_lr.
        # We also apply force shifting to reduce discontinuity artifacts.
        if self.cutoff_lr is not None:
            # Calculate electrostatic energies per long-range edge
            atomic_electrostatic_energy_ij = coulomb_erf_shifted_force_smooth(
                partial_charges,
                d_ij_lr,
                idx_i_lr,
                idx_j_lr,
                ke=self.ke,
                sigma=self.electrostatic_energy_scale,
                cutoff=self.cutoff_lr,
                cuton=self.cutoff_lr * 0.45,
                neighborlist_format=self.neighborlist_format
            )
        # If no cutoff is set, we just apply damping with error function and no explicit smoothing to zero.
        else:
            # Calculate electrostatic energies per long-range edge
            atomic_electrostatic_energy_ij = coulomb_erf(
                partial_charges,
                d_ij_lr,
                idx_i_lr,
                idx_j_lr,
                ke=self.ke,
                sigma=self.electrostatic_energy_scale,
                neighborlist_format=self.neighborlist_format
            )

        # Calculate electrostatic atomic energies via summing over long-range neighbors
        atomic_electrostatic_energy = segment_sum(
            atomic_electrostatic_energy_ij,
            segment_ids=idx_i_lr,
            num_segments=num_nodes
        )  # (num_nodes)

        # Mask padded nodes
        atomic_electrostatic_energy = safe_scale(atomic_electrostatic_energy, node_mask)

        return dict(electrostatic_energy=atomic_electrostatic_energy)

    def reset_output_convention(self, output_convention):
        pass


class DispersionEnergySparse(nn.Module):
    prop_keys: Dict
    cutoff_lr: float
    cutoff_lr_damping: float
    hirshfeld_ratios: Optional[Any]
    dispersion_energy_scale: float = 1.0

    neighborlist_format: str = 'sparse'  # or 'ordered_sparse'
    module_name = 'dispersion_energy'

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        node_mask = inputs['node_mask']  # (num_nodes)
        num_nodes = len(node_mask)
        idx_i_lr = inputs['idx_i_lr']
        idx_j_lr = inputs['idx_j_lr']
        d_ij_lr = inputs['d_ij_lr']

        # Determine input dtype
        input_dtype = d_ij_lr.dtype

        # Calculate Hirshfeld ratios
        hirshfeld_ratios = self.hirshfeld_ratios(inputs)['hirshfeld_ratios']

        # Get atomic numbers (needed to link to the free-atom reference values)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)

        # Calculate alpha_ij and C6_ij using mixing rules
        alpha_ij, C6_ij = mixing_rules(
            atomic_numbers,
            idx_i_lr,
            idx_j_lr,
            hirshfeld_ratios
        )

        # Use cubic fit for gamma
        gamma_ij = gamma_cubic_fit(alpha_ij)

        # Get dispersion energy, positions are converted to to a.u.
        dispersion_energy_ij = vdw_QDO_disp_damp(
            d_ij_lr / jnp.asarray(Bohr, dtype=input_dtype),
            gamma_ij,
            C6_ij,
            alpha_ij,
            jnp.asarray(self.dispersion_energy_scale, dtype=input_dtype),
            self.neighborlist_format
        )

        # If long-range cutoff is given, one needs to damp dispersion smoothly to zero at cutoff_lr.
        if self.cutoff_lr is not None:
            if self.cutoff_lr_damping is None:
                raise ValueError(
                    f"cutoff_lr is but cutoff_lr_damping is not set. "
                    f"received {self.cutoff_lr=} and {self.cutoff_lr_damping=}."
                )

            w = safe_mask(
                d_ij_lr > 0,
                partial(switching_fn, x_on=self.cutoff_lr - self.cutoff_lr_damping, x_off=self.cutoff_lr),
                d_ij_lr,
                0.
            )

            dispersion_energy_ij = safe_scale(dispersion_energy_ij, w, 0.)

        atomic_dispersion_energy = segment_sum(
            dispersion_energy_ij,
            segment_ids=idx_i_lr,
            num_segments=num_nodes
        )  # (num_nodes)

        atomic_dispersion_energy = safe_scale(atomic_dispersion_energy, node_mask)

        return dict(dispersion_energy=atomic_dispersion_energy)

    def reset_output_convention(self, output_convention):
        pass
