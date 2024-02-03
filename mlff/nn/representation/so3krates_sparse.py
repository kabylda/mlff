import flax.linen as nn
import jax
from mlff.nn.stacknet import StackNetSparse
from mlff.nn.embed import GeometryEmbedSparse, AtomTypeEmbedSparse
from mlff.nn.layer import SO3kratesLayerSparse
from mlff.nn.observable import EnergySparse, DipoleSparse, HirshfeldSparse
from typing import Sequence


def init_so3krates_sparse(
        num_layers: int = 2,
        num_features: int = 128,
        num_heads: int = 4,
        num_features_head: int = 32,
        radial_basis_fn: str = 'bernstein',
        num_radial_basis_fn: int = 16,
        cutoff_fn: str = 'exponential',
        cutoff: float = 5.,
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
        energy_regression_dim: int = 128,
        energy_activation_fn: str = 'identity',
        energy_learn_atomic_type_scales: bool = False,
        energy_learn_atomic_type_shifts: bool = False,
        input_convention: str = 'positions'
):
    atom_type_embed = AtomTypeEmbedSparse(
        num_features=num_features,
        prop_keys=None
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

    energy = EnergySparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        learn_atomic_type_scales=energy_learn_atomic_type_scales,
        learn_atomic_type_shifts=energy_learn_atomic_type_shifts,
    )

    dipole = DipoleSparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        # learn_atomic_type_scales=energy_learn_atomic_type_scales,
        # learn_atomic_type_shifts=energy_learn_atomic_type_shifts,
    )

    hirshfeld_ratios = HirshfeldSparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        # learn_atomic_type_scales=energy_learn_atomic_type_scales,
        # learn_atomic_type_shifts=energy_learn_atomic_type_shifts,
    )    

    return StackNetSparse(
        geometry_embeddings=[geometry_embed],
        feature_embeddings=[atom_type_embed],
        layers=layers,
        observables=[energy, dipole, hirshfeld_ratios],
        prop_keys=None
    )
