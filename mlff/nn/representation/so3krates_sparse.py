import flax.linen as nn
from mlff.nn.stacknet import StackNetSparse
from mlff.nn.embed import GeometryEmbedSparse
from mlff.nn.layer import SO3kratesLayerSparse
from mlff.nn.observable import EnergySparse
from mlff.nn.observable import ElectrostaticEnergySparse
from mlff.nn.observable import DispersionEnergySparse
from mlff.nn.observable import ZBLRepulsionSparse
from mlff.nn.observable import DipoleVecSparse
from mlff.nn.observable import HirshfeldSparse
from mlff.nn.observable import PartialChargesSparse

from typing import Optional, Sequence

from .representation_utils import make_embedding_modules


def init_so3krates_sparse(
        num_layers: int = 2,
        num_features: int = 128,
        num_heads: int = 4,
        num_features_head: int = 32,
        radial_basis_fn: str = 'bernstein',
        num_radial_basis_fn: int = 16,
        cutoff_fn: str = 'exponential',
        cutoff: float = 5.,
        cutoff_lr: Optional[float] = None,
        degrees: Sequence[int] = [1, 2, 3, 4],
        residual_mlp_1: bool = True,
        residual_mlp_2: bool = True,
        layer_normalization_1: bool = False,
        layer_normalization_2: bool = False,
        message_normalization: str = 'sqrt_num_features',
        avg_num_neighbors: float = None,
        qk_non_linearity: str = 'silu',
        activation_fn: str = 'silu',
        layers_behave_like_identity_fn_at_init: bool = False,
        output_is_zero_at_init: bool = True,
        use_charge_embed: bool = False,
        use_spin_embed: bool = False,
        energy_regression_dim: int = 128,
        energy_activation_fn: str = 'identity',
        energy_learn_atomic_type_scales: bool = False,
        energy_learn_atomic_type_shifts: bool = False,
        electrostatic_energy_bool: bool = False,
        electrostatic_energy_scale: float = 1.0,
        dispersion_energy_bool: bool = False,
        dispersion_energy_cutoff_lr_damping: Optional[float] = None,
        dispersion_energy_scale: float = 1.0,
        zbl_repulsion_bool: bool = False,
        use_final_bias_bool: bool = True,
        return_representations_bool: bool = False,
        input_convention: str = 'positions',
        neighborlist_format_lr: str = 'sparse'  # or 'ordered_sparse'
):
    embedding_modules = make_embedding_modules(
        num_features=num_features,
        use_spin_embed=use_spin_embed,
        use_charge_embed=use_charge_embed
    )

    geometry_embed = GeometryEmbedSparse(
        degrees=degrees,
        radial_basis_fn=radial_basis_fn,
        num_radial_basis_fn=num_radial_basis_fn,
        cutoff_fn=cutoff_fn,
        cutoff=cutoff,
        input_convention=input_convention,
        prop_keys=None
    )

    layers = [SO3kratesLayerSparse(
        degrees=degrees,
        use_spherical_filter=True,
        num_heads=num_heads,
        num_features_head=num_features_head,
        qk_non_linearity=getattr(nn.activation, qk_non_linearity) if qk_non_linearity != 'identity' else lambda u: u,
        residual_mlp_1=residual_mlp_1,
        residual_mlp_2=residual_mlp_2,
        layer_normalization_1=layer_normalization_1,
        layer_normalization_2=layer_normalization_2,
        message_normalization=message_normalization,
        avg_num_neighbors=avg_num_neighbors,
        activation_fn=getattr(nn.activation, activation_fn) if activation_fn != 'identity' else lambda u: u,
        behave_like_identity_fn_at_init=layers_behave_like_identity_fn_at_init
    ) for i in range(num_layers)]

    partial_charges = PartialChargesSparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
    )

    hirshfeld_ratios = HirshfeldSparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
    ) 

    dispersion_energy = DispersionEnergySparse(
        prop_keys=None,
        hirshfeld_ratios=hirshfeld_ratios,
        cutoff_lr=cutoff_lr,
        cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
        dispersion_energy_scale=dispersion_energy_scale,
        neighborlist_format=neighborlist_format_lr
    )

    electrostatic_energy = ElectrostaticEnergySparse(
        prop_keys=None,
        partial_charges=partial_charges,
        cutoff_lr=cutoff_lr,
        electrostatic_energy_scale=electrostatic_energy_scale,
        neighborlist_format=neighborlist_format_lr
    )

    dipole_vec = DipoleVecSparse(
        prop_keys=None,
        partial_charges=partial_charges
    )

    zbl_repulsion = ZBLRepulsionSparse(
        prop_keys=None
    )
    
    energy = EnergySparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        learn_atomic_type_scales=energy_learn_atomic_type_scales,
        learn_atomic_type_shifts=energy_learn_atomic_type_shifts,
        electrostatic_energy=electrostatic_energy,
        dispersion_energy=dispersion_energy,
        zbl_repulsion=zbl_repulsion,
        electrostatic_energy_bool=electrostatic_energy_bool,
        dispersion_energy_bool=dispersion_energy_bool,
        zbl_repulsion_bool=zbl_repulsion_bool,
        use_final_bias_bool=use_final_bias_bool
    )

    return StackNetSparse(
        geometry_embeddings=[geometry_embed],
        feature_embeddings=embedding_modules,
        layers=layers,
        observables=[energy, dipole_vec, hirshfeld_ratios],
        return_representations_bool=return_representations_bool,
        prop_keys=None
    )
