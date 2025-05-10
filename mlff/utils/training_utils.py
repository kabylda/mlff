import clu.metrics as clu_metrics
import jraph
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb

from flax import traverse_util
from flax.core.frozen_dict import unfreeze
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from ..utils import gradient_utils
from ..utils import checkpoint_utils

from robust_loss_jax import distribution as robust_loss_dist
# def interpolate1d(x, values, tangents):
#   r"""Perform cubic hermite spline interpolation on a 1D spline.

#   The x coordinates of the spline knots are at [0 : len(values)-1].
#   Queries outside of the range of the spline are computed using linear
#   extrapolation. See https://en.wikipedia.org/wiki/Cubic_Hermite_spline
#   for details, where "x" corresponds to `x`, "p" corresponds to `values`, and
#   "m" corresponds to `tangents`.

#   Args:
#     x: A tensor containing the set of values to be used for interpolation into
#       the spline.
#     values: A vector containing the value of each knot of the spline being
#       interpolated into. Must be the same length as `tangents`.
#     tangents: A vector containing the tangent (derivative) of each knot of the
#       spline being interpolated into. Must be the same length as `values` and
#       the same type as `x`.

#   Returns:
#     The result of interpolating along the spline defined by `values`, and
#     `tangents`, using `x` as the query values. Will be the same shape as `x`.
#   """
#   assert len(values.shape) == 1
#   assert len(tangents.shape) == 1
#   assert values.shape[0] == tangents.shape[0]

#   # Find the indices of the knots below and above each x.
#   x_lo = jnp.int32(jnp.floor(jnp.clip(x, 0., values.shape[0] - 2)))
#   x_hi = x_lo + 1

#   # Compute the relative distance between each `x` and the knot below it.
#   t = x - x_lo

#   # Compute the cubic hermite expansion of `t`.
#   t_sq = t**2
#   t_cu = t * t_sq
#   h01 = -2 * t_cu + 3 * t_sq
#   h00 = 1 - h01
#   h11 = t_cu - t_sq
#   h10 = h11 - t_sq + t

#   # Linearly extrapolate above and below the extents of the spline for all
#   # values.
#   value_before = tangents[0] * t + values[0]
#   value_after = tangents[-1] * (t - 1) + values[-1]

#   # Cubically interpolate between the knots below and above each query point.
#   neighbor_values_lo = jnp.take(values, x_lo)
#   neighbor_values_hi = jnp.take(values, x_hi)
#   neighbor_tangents_lo = jnp.take(tangents, x_lo)
#   neighbor_tangents_hi = jnp.take(tangents, x_hi)

#   value_mid = (
#       neighbor_values_lo * h00 + neighbor_values_hi * h01 +
#       neighbor_tangents_lo * h10 + neighbor_tangents_hi * h11)

#   # Return the interpolated or extrapolated values for each query point,
#   # depending on whether or not the query lies within the span of the spline.
#   return jnp.where(t < 0., value_before,
#                    jnp.where(t > 1., value_after, value_mid))

# @jax.custom_jvp
# def fake_clip(a, a_min, a_max):
#   """jnp.clip() but the gradient doesn't get clipped on the backward pass."""
#   return jnp.clip(a, a_min, a_max)


# @fake_clip.defjvp
# def fake_clip_jvp(primals, tangents):
#   """Override fake_clip()'s gradient so that it's a no-op."""
#   return jnp.clip(*primals), tangents[0]


# @jax.jit
# def lossfun(x, alpha, scale):
#   r"""Implements the general form of the loss.

#   This implements the rho(x, \alpha, c) function described in "A General and
#   Adaptive Robust Loss Function", Jonathan T. Barron,
#   https://arxiv.org/abs/1701.03077.

#   Args:
#     x: The residual for which the loss is being computed. x can have any shape,
#       and alpha and scale will be broadcasted to match x's shape if necessary.
#     alpha: The shape parameter of the loss (\alpha in the paper), where more
#       negative values produce a loss with more robust behavior (outliers "cost"
#       less), and more positive values produce a loss with less robust behavior
#       (outliers are penalized more heavily). Alpha can be any value in
#       [-infinity, infinity], but the gradient of the loss with respect to alpha
#       is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
#       interpolation between several discrete robust losses:
#         alpha=-Infinity: Welsch/Leclerc Loss.
#         alpha=-2: Geman-McClure loss.
#         alpha=0: Cauchy/Lortentzian loss.
#         alpha=1: Charbonnier/pseudo-Huber loss.
#         alpha=2: L2 loss.
#     scale: The scale parameter of the loss. When |x| < scale, the loss is an
#       L2-like quadratic bowl, and when |x| > scale the loss function takes on a
#       different shape according to alpha.

#   Returns:
#     The losses for each element of x, in the same shape as x.
#   """
#   eps = jnp.finfo(jnp.float32).eps
#   maxval = 1e15

#   # A "safe" versions of expm1 that will not NaN-out on large inputs.
#   expm1_safe = lambda x: jnp.expm1(jnp.minimum(x, 43))

#   # `scale` must be > 0.
#   scale = jnp.maximum(eps, scale)

#   # Large values of |x| can cause non-finite gradients.
#   x = fake_clip(x, -maxval, maxval)

#   # The loss when alpha == 2. This will get reused repeatedly.
#   loss_two = 0.5 * (x / scale)**2

#   # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
#   a = jnp.where(alpha >= 0, jnp.ones_like(alpha),
#                 -jnp.ones_like(alpha)) * jnp.maximum(eps, jnp.abs(alpha))

#   # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
#   b = jnp.maximum(eps, jnp.abs(a - 2))

#   # The loss when not in one of the special casess.
#   loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1)**(0.5 * a) - 1)

#   # Select which of the cases of the loss to return as a function of alpha.
#   return jnp.where(
#       alpha == -jnp.inf, -expm1_safe(-loss_two),
#       jnp.where(
#           alpha == 0, jnp.log1p(loss_two),
#           jnp.where(alpha == 2, loss_two,
#                     jnp.where(alpha == jnp.inf, expm1_safe(loss_two),
#                               loss_ow))))

# import jax.random as random
# #from robust_loss_jax import cubic_spline
# #from robust_loss_jax import general


# def get_resource_as_file(path):
#   """A uniform interface for internal/open-source files."""

#   class NullContextManager(object):

#     def __init__(self, dummy_resource=None):
#       self.dummy_resource = dummy_resource

#     def __enter__(self):
#       return self.dummy_resource

#     def __exit__(self, *args):
#       pass

#   #return NullContextManager('./' + path)
#   return NullContextManager(path)


# def get_resource_filename(path):
#   """A uniform interface for internal/open-source filenames."""
#   #return './' + path
#   return path


# def partition_spline_curve(alpha):
#   """Applies a curve to alpha >= 0 to compress its range before interpolation.

#   This is a weird hand-crafted function designed to take in alpha values and
#   curve them to occupy a short finite range that works well when using spline
#   interpolation to model the partition function Z(alpha). Because Z(alpha)
#   is only varied in [0, 4] and is especially interesting around alpha=2, this
#   curve is roughly linear in [0, 4] with a slope of ~1 at alpha=0 and alpha=4
#   but a slope of ~10 at alpha=2. When alpha > 4 the curve becomes logarithmic.
#   Some (input, output) pairs for this function are:
#     [(0, 0), (1, ~1.2), (2, 4), (3, ~6.8), (4, 8), (8, ~8.8), (400000, ~12)]
#   This function is continuously differentiable.

#   Args:
#     alpha: A tensor with values >= 0.

#   Returns:
#     A tensor of curved values >= 0 with the same type as `alpha`, to be
#     used as input x-coordinates for spline interpolation.
#   """
#   log_safe = lambda z: jnp.log(jnp.minimum(z, 3e37))
#   x = jnp.where(alpha < 4,
#                 (2.25 * alpha - 4.5) / (jnp.abs(alpha - 2) + 0.25) + alpha + 2,
#                 (5 / 18) * log_safe(4 * alpha - 15) + 8)
#   return x


# def inv_partition_spline_curve(x):
#   """The inverse of partition_spline_curve()."""
#   exp_safe = lambda z: jnp.exp(jnp.minimum(z, 87.5))
#   alpha = jnp.where(
#       x < 8,
#       0.5 * x + jnp.where(x <= 4, 1.25 - jnp.sqrt(1.5625 - x + .25 * x**2),
#                           -1.25 + jnp.sqrt(9.5625 - 3 * x + .25 * x**2)),
#       3.75 + 0.25 * exp_safe(x * 3.6 - 28.8))
#   return alpha


# class Distribution(object):
#   """A wrapper class around the distribution."""

#   def __init__(self):
#     """Initialize the distribution.

#     Load the values, tangents, and x-coordinate scaling of a spline that
#     approximates the partition function. The spline was produced by running
#     the script in fit_partition_spline.py.
#     """
#     with get_resource_as_file(
#         '/mnt/tier2/project/p200717/kabylda/so3lr_train_qcml/so3lr_v0.1.0/lib/python3.12/site-packages/mlff/utils/partition_spline.npz') as spline_file:
#       with jnp.load(spline_file, allow_pickle=False) as f:
#         self._spline_x_scale = f['x_scale']
#         self._spline_values = f['values']
#         self._spline_tangents = f['tangents']

#   def log_base_partition_function(self, alpha):
#     r"""Approximate the distribution's log-partition function with a 1D spline.

#     Because the partition function (Z(\alpha) in the paper) of the distribution
#     is difficult to model analytically, we approximate it with a (transformed)
#     cubic hermite spline: Each alpha is pushed through a nonlinearity before
#     being used to interpolate into a spline, which allows us to use a relatively
#     small spline to accurately model the log partition function over the range
#     of all non-negative input values.

#     Args:
#       alpha: A tensor containing the set of alphas for which we would like an
#         approximate log partition function. Must be non-negative, as the
#         partition function is undefined when alpha < 0.

#     Returns:
#       An approximation of log(Z(alpha)) accurate to within 1e-6
#     """
#     # The partition function is undefined when `alpha`< 0.
#     alpha = jnp.maximum(0, alpha)
#     # Transform `alpha` to the form expected by the spline.
#     x = partition_spline_curve(alpha)
#     # Interpolate into the spline.
#     return interpolate1d(x * self._spline_x_scale,
#                                       self._spline_values,
#                                       self._spline_tangents)

#   def nllfun(self, x, alpha, scale):
#     r"""Implements the negative log-likelihood (NLL).

#     Specifically, we implement -log(p(x | 0, \alpha, c) of Equation 16 in the
#     paper as nllfun(x, alpha, shape).

#     Args:
#       x: The residual for which the NLL is being computed. x can have any shape,
#         and alpha and scale will be broadcasted to match x's shape if necessary.
#         Must be a tensorflow tensor or numpy array of floats.
#       alpha: The shape parameter of the NLL (\alpha in the paper), where more
#         negative values cause outliers to "cost" more and inliers to "cost"
#         less. Alpha can be any non-negative value, but the gradient of the NLL
#         with respect to alpha has singularities at 0 and 2 so you may want to
#         limit usage to (0, 2) during gradient descent. Must be a tensorflow
#         tensor or numpy array of floats. Varying alpha in that range allows for
#         smooth interpolation between a Cauchy distribution (alpha = 0) and a
#         Normal distribution (alpha = 2) similar to a Student's T distribution.
#       scale: The scale parameter of the loss. When |x| < scale, the NLL is like
#         that of a (possibly unnormalized) normal distribution, and when |x| >
#         scale the NLL takes on a different shape according to alpha. Must be a
#         tensorflow tensor or numpy array of floats.

#     Returns:
#       The NLLs for each element of x, in the same shape as x. This is returned
#       as a TensorFlow graph node of floats with the same precision as x.
#     """
#     alpha = jnp.maximum(0, alpha)
#     scale = jnp.maximum(jnp.finfo(jnp.float32).eps, scale)
#     loss = lossfun(x, alpha, scale)
#     return loss + jnp.log(scale) + self.log_base_partition_function(alpha)

#   def draw_samples(self, rng, alpha, scale):
#     r"""Draw samples from the robust distribution.

#     This function implements Algorithm 1 the paper. This code is written to
#     allow for sampling from a set of different distributions, each parametrized
#     by its own alpha and scale values, as opposed to the more standard approach
#     of drawing N samples from the same distribution. This is done by repeatedly
#     performing N instances of rejection sampling for each of the N distributions
#     until at least one proposal for each of the N distributions has been
#     accepted. All samples assume a zero mean --- to get non-zero mean samples,
#     just add each mean to each sample.

#     Args:
#       rng: A JAX pseudo random number generated, from random.PRNG().
#       alpha: A tensor where each element is the shape parameter of that
#         element's distribution. Must be > 0.
#       scale: A tensor where each element is the scale parameter of that
#         element's distribution. Must be >=0 and the same shape as `alpha`.

#     Returns:
#       A tensor with the same shape as `alpha` and `scale` where each element is
#       a sample drawn from the zero-mean distribution specified for that element
#       by `alpha` and `scale`.
#     """
#     assert jnp.all(scale > 0)
#     assert jnp.all(alpha >= 0)
#     assert jnp.all(jnp.array(alpha.shape) == jnp.array(scale.shape))
#     shape = alpha.shape

#     samples = jnp.zeros(shape)
#     accepted = jnp.zeros(shape, dtype=bool)

#     # Rejection sampling.
#     while not jnp.all(accepted):

#       # The sqrt(2) scaling of the Cauchy distribution corrects for our
#       # differing conventions for standardization.
#       rng, key = random.split(rng)
#       cauchy_sample = random.cauchy(key, shape=shape) * jnp.sqrt(2)

#       # Compute the likelihood of each sample under its target distribution.
#       nll = self.nllfun(cauchy_sample, alpha, 1)

#       # Bound the NLL. We don't use the approximate loss as it may cause
#       # unpredictable behavior in the context of sampling.
#       nll_bound = (
#           lossfun(cauchy_sample, 0, 1) +
#           self.log_base_partition_function(alpha))

#       # Draw N samples from a uniform distribution, and use each uniform
#       # sample to decide whether or not to accept each proposal sample.
#       rng, key = random.split(rng)
#       uniform_sample = random.uniform(key, shape=shape)
#       accept = uniform_sample <= jnp.exp(nll_bound - nll)

#       # If a sample is accepted, replace its element in `samples` with the
#       # proposal sample, and set its bit in `accepted` to True.
#       samples = jnp.where(accept, cauchy_sample, samples)
#       accepted = accept | accepted

#     # Because our distribution is a location-scale family, we sample from
#     # p(x | 0, \alpha, 1) and then scale each sample by `scale`.
#     samples *= scale

#     return samples

# Initialize the distribution object for adaptive robust loss
ROBUST_LOSS_DIST = robust_loss_dist.Distribution()
# ROBUST_LOSS_DIST = Distribution()

def print_metrics(epoch, eval_metrics):
    formatted_output = f"{epoch}: "
    for key, value in eval_metrics.items():
        if isinstance(value, np.ndarray) and value.size == 1:
            formatted_output += f"{key}={value.item():.4f}, "
        else:
            formatted_output += f"{key}={', '.join(map('{:.4f}'.format, value))}, " if isinstance(value, np.ndarray) else f"{key}={value:.4f}, "
    return formatted_output.rstrip(", ")


def graph_mse_loss(y, y_label, batch_segments, graph_mask, scale, use_robust_loss=False, robust_loss_alpha=1.99):
    del batch_segments

    assert y.shape == y_label.shape

    full_mask = ~jnp.isnan(
        y_label
    ) & jnp.expand_dims(
        graph_mask, [y_label.ndim - 1 - o for o in range(0, y_label.ndim - 1)]
    )
    denominator = full_mask.sum().astype(y.dtype)
    
    if use_robust_loss:
        # Adaptive robust loss
        diff = jnp.where(full_mask, y - y_label, 0).reshape(-1)
        loss = jnp.sum(2 * scale * ROBUST_LOSS_DIST.nllfun(diff, robust_loss_alpha, 1.0)) / denominator
    else:
        # Regular L2 loss
        loss = (
                jnp.sum(
                    2 * scale * optax.l2_loss(
                        jnp.where(full_mask, y, 0).reshape(-1),
                        jnp.where(full_mask, y_label, 0).reshape(-1),
                    )
                )
                / denominator
        )
    return loss


def node_mse_loss(y, y_label, batch_segments, graph_mask, scale, use_robust_loss=False, robust_loss_alpha=1.99):
    assert y.shape == y_label.shape

    num_graphs = graph_mask.sum().astype(y.dtype)  # ()

    if use_robust_loss:
        # Adaptive robust loss
        diff = y - y_label
        masked_diff = gradient_utils.safe_mask(
            fn=lambda u: u,
            operand=diff,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )
        
        squared = gradient_utils.safe_mask(
            fn=lambda u: 2 * ROBUST_LOSS_DIST.nllfun(u, robust_loss_alpha, 1.0),
            operand=masked_diff,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )
    else:
        # Regular L2 loss
        squared = gradient_utils.safe_mask(
            fn=lambda u: jnp.square(u),
            operand=y - y_label,
            mask=~jnp.isnan(y_label),
            placeholder=0.
        )

    # sum up the losses for node properties along the non-leading dimension. For e.g. scalar node quantities
    # this does not have any effect, but e.g. for vectorial and tensorial node properties one averages over all
    # additional non-leading dimension. E.g. for forces this corresponds to taking mean over x, y, z component.
    node_mean_squared = squared.reshape(len(squared), -1).mean(axis=-1)  # (num_nodes)

    per_graph_mse = jraph.segment_mean(
        data=node_mean_squared,
        segment_ids=batch_segments,
        num_segments=len(graph_mask)
    )  # (num_graphs)

    # Set contributions from padding graphs to zero.
    per_graph_mse = jnp.where(
        graph_mask,
        per_graph_mse,
        jnp.asarray(0., dtype=per_graph_mse.dtype)
    )  # (num_graphs)

    # Create msk that has True when data is present and is false if no data is present, i.e. y_label equals NaN.
    # Note that padding graphs still have zero valued entries.
    data_msk = ~jnp.isnan(
        jax.ops.segment_max(
            data=jnp.max(y_label.reshape(len(y_label), -1), axis=-1),
            segment_ids=batch_segments,
            num_segments=len(graph_mask)
        )  # evaluates to NaN if one entry in the segment is NaN.
    )  # (num_graphs)

    # Set contributions from graphs for which no node labels are present to zero.
    per_graph_mse = jnp.where(
        data_msk,
        per_graph_mse,
        jnp.asarray(0., dtype=per_graph_mse.dtype)
    )  # (num_graphs)

    # Calculate the number of graphs that have no data present.
    num_graphs_no_data = jnp.where(
        data_msk,
        jnp.asarray(0., dtype=per_graph_mse.dtype),
        jnp.asarray(1., dtype=per_graph_mse.dtype),
    ).sum()

    # subtract the number of graphs for which no data is present.
    num_graphs = num_graphs - num_graphs_no_data

    # Calculate mean and scale. Prevent the case of division by zero if no data is present at all.
    mse = scale * jnp.sum(per_graph_mse) / jnp.maximum(num_graphs, 1.)  # ()

    return mse


def graph_mae_loss(y, y_label, batch_segments, graph_mask, scale):
    assert y.shape == y_label.shape

    full_mask = ~jnp.isnan(
        y_label
    ) & jnp.expand_dims(
        graph_mask, [y_label.ndim - 1 - o for o in range(0, y_label.ndim - 1)]
    )
    denominator = full_mask.sum().astype(y.dtype)
    
    # Calculate absolute error instead of squared error
    loss = (
            jnp.sum(
                jnp.abs(
                    jnp.where(full_mask, y, 0).reshape(-1) - 
                    jnp.where(full_mask, y_label, 0).reshape(-1)
                )
            ) 
            / denominator
    )
    return loss


def node_mae_loss(y, y_label, batch_segments, graph_mask, scale):
    assert y.shape == y_label.shape

    num_graphs = graph_mask.sum().astype(y.dtype)  # ()

    # Use absolute error for MAE
    abs_error = gradient_utils.safe_mask(
        fn=lambda u: jnp.abs(u),
        operand=y - y_label,
        mask=~jnp.isnan(y_label),
        placeholder=0.
    )

    # sum up the losses for node properties along the non-leading dimension
    node_mean_abs = abs_error.reshape(len(abs_error), -1).mean(axis=-1)  # (num_nodes)

    per_graph_mae = jraph.segment_mean(
        data=node_mean_abs,
        segment_ids=batch_segments,
        num_segments=len(graph_mask)
    )  # (num_graphs)

    # Set contributions from padding graphs to zero.
    per_graph_mae = jnp.where(
        graph_mask,
        per_graph_mae,
        jnp.asarray(0., dtype=per_graph_mae.dtype)
    )  # (num_graphs)

    # Create mask that has True when data is present and is false if no data is present
    data_msk = ~jnp.isnan(
        jax.ops.segment_max(
            data=jnp.max(y_label.reshape(len(y_label), -1), axis=-1),
            segment_ids=batch_segments,
            num_segments=len(graph_mask)
        )  # evaluates to NaN if one entry in the segment is NaN.
    )  # (num_graphs)

    # Set contributions from graphs for which no node labels are present to zero.
    per_graph_mae = jnp.where(
        data_msk,
        per_graph_mae,
        jnp.asarray(0., dtype=per_graph_mae.dtype)
    )  # (num_graphs)

    # Calculate the number of graphs that have no data present.
    num_graphs_no_data = jnp.where(
        data_msk,
        jnp.asarray(0., dtype=per_graph_mae.dtype),
        jnp.asarray(1., dtype=per_graph_mae.dtype),
    ).sum()

    # subtract the number of graphs for which no data is present.
    num_graphs = num_graphs - num_graphs_no_data

    # Calculate mean. Prevent division by zero if no data is present.
    mae = jnp.sum(per_graph_mae) / jnp.maximum(num_graphs, 1.)
    
    return mae

property_to_mae = {
    'energy': graph_mae_loss,
    'stress': graph_mae_loss,
    'forces': node_mae_loss,
    'dipole_vec': graph_mae_loss,
    'hirshfeld_ratios': node_mae_loss,
}

property_to_loss = {
    'energy': graph_mse_loss,
    'stress': graph_mse_loss,
    'forces': node_mse_loss,
    'dipole_vec': graph_mse_loss,
    'hirshfeld_ratios': node_mse_loss,
}


def make_loss_fn(obs_fn: Callable, weights: Dict, scales: Dict = None, 
                 use_robust_loss: bool = False, robust_loss_alpha: float = 1.99):
    # Targets are collected based on the loss weights.
    targets = list(weights.keys())

    if scales is None:
        _scales = {k: jnp.ones(1) for k in targets}
    else:
        _scales = scales

    @jax.jit
    def loss_fn(params, batch: Dict[str, jnp.ndarray]):
        # Everything that is not a target is a input.
        inputs = {k: v for k, v in batch.items() if k not in targets}

        # Collect the targets.
        outputs_true = {k: v for k, v in batch.items() if k in targets}

        # Make predictions.
        outputs_predict = obs_fn(params, **inputs)
        loss = jnp.zeros(1)
        metrics = {}
        # Iterate over the targets, calculate loss and multiply with loss weights and scales.
        for target in targets:
            target_loss_fn = property_to_loss[target]
            _l = target_loss_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
                use_robust_loss=use_robust_loss,
                robust_loss_alpha=robust_loss_alpha
            )

            loss += weights[target] * _l
            metrics.update({f'{target}_mse': _l / _scales[target].mean()})

        loss = jnp.reshape(loss, ())
        metrics.update({'loss': loss})

        return loss, metrics

    return loss_fn


def make_val_fn(obs_fn: Callable, weights: Dict, scales: Dict = None):
    """Creates a validation function that calculates MAE metrics
    
    Args:
        obs_fn (Callable): Observable function that returns predicted properties
        weights (Dict): Dictionary of property names and their weights
        scales (Dict, optional): Dictionary of scales for each property. Defaults to None.
    
    Returns:
        Callable: Validation function that returns MAE metrics
    """
    # Targets are collected based on the loss weights
    targets = list(weights.keys())

    if scales is None:
        _scales = {k: jnp.ones(1) for k in targets}
    else:
        _scales = scales

    @jax.jit
    def val_fn(params, batch: Dict[str, jnp.ndarray]):
        # Everything that is not a target is a input
        inputs = {k: v for k, v in batch.items() if k not in targets}

        # Collect the targets
        outputs_true = {k: v for k, v in batch.items() if k in targets}

        # Make predictions
        outputs_predict = obs_fn(params, **inputs)
        metrics = {}
        
        # Iterate over the targets and calculate MAE metrics
        for target in targets:
            target_mae_fn = property_to_mae[target]
            _mae = target_mae_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask']
            )

            metrics.update({f'{target}_mae': _mae})

            # Calculate MSE metrics
            target_loss_fn = property_to_loss[target]
            _mse = target_loss_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
                use_robust_loss=False
            )
            metrics.update({f'{target}_mse': _mse / _scales[target].mean()})

        # Calculate total loss using MSE for compatibility with existing code
        loss = jnp.zeros(1)
        loss_mae = jnp.zeros(1)
        for target in targets:
            loss += weights[target] * metrics[f'{target}_mse'] * _scales[target].mean()
            loss_mae += weights[target] * metrics[f'{target}_mae'] * _scales[target].mean()
        
        metrics.update({'loss': loss})
        metrics.update({'loss_mae': loss_mae})

        return loss, metrics

    return val_fn


def make_training_step_fn(
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        log_gradient_values: bool
):
    """
    Make a training step fn, which takes params, optimizer state, and a batch of data and returns
    new params based on the gradients according to the loss_fn, new optimizer state and metrics.

    Args:
        optimizer (optax.GradientTransformation): Optax optimizer.
        loss_fn (Callable): Loss function.
        log_gradient_values (bool): Log gradient values for each leaf in the params pytree.

    Returns:
        Training step fn.

    """

    @jax.jit
    def training_step_fn(
            params,
            opt_state,
            batch
    ):
        """
        Training step.

        Args:
            params (FrozenDict): Parameter dictionary.
            opt_state: Optax optimizer state.
            batch (Tuple): Batch of validation data.

        Returns:
            Updated state and metrics.

        """
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True
        )(
            params,
            batch
        )

        if log_gradient_values:
            metrics['grad_norm'] = unfreeze(jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1), axis=0), grads))

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params
        )

        params = optax.apply_updates(
            params=params,
            updates=updates
        )

        metrics['grad_norm'] = optax.global_norm(grads)

        return params, opt_state, metrics

    return training_step_fn


def make_validation_step_fn(
        metric_fn: Callable
):
    """
    Make validation step function, which takes params and batch of data as input and returns metrics.

    Args:
        metric_fn (Callable): Function that calculates metrics, given params and batch of data.

    Returns:
        Validation step function.

    """
    @jax.jit
    def validation_step_fn(params, batch) -> Dict[str, jnp.ndarray]:
        """
        Validation step.

        Args:
            params (FrozenDict): Parameters.
            batch (Tuple): Batch of validation data.

        Returns:
            Validation metrics.
        """
        _, metrics = metric_fn(
            params,
            batch
        )

        return metrics

    return validation_step_fn


def fit(
        model,
        optimizer,
        loss_fn,
        graph_to_batch_fn,
        training_data,
        validation_data,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs,
        batch_max_num_pairs,
        params=None,
        val_fn=None,
        num_epochs: int = 100,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart: bool = False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        log_gradient_values: bool = False
):
    """
    Fit model.

    Args:
        model: flax module.
        optimizer: optax optimizer.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        graph_to_batch_fn (Callable): Function that takes a batched graph and returns a batch for the loss_fn.
        training_data (Sequence): Sequence of jraph.GraphTuples.
        validation_data (Sequence): Sequence of jraph.GraphTuples.
        batch_max_num_nodes (int): Maximal number of nodes per batch.
        batch_max_num_edges (int): Maximal number of edges per batch.
        batch_max_num_graphs (int): Maximal number of graphs per batch.
        batch_max_num_pairs (int): Maximal number of pairs in long-range indices.
        params: Parameters to start from during training. If not given, either new parameters are initialized randomly
            or loaded from ckpt_dir if the checkpoint already exists and `allow_restart=True`.
        val_fn (Callable, optional): Validation function to use for metrics during validation. 
            If None, loss_fn will be used. Defaults to None.
        num_epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_num_steps (int): Evaluate the metrics every num-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias. If true, wandb.init() must be called before call to fit().
        log_gradient_values (bool): Gradient values for each set of weights is logged.
    Returns:

    """
    numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(exist_ok=True)

    # Create orbax CheckpointManager.
    if ckpt_manager_options is None:
        ckpt_manager_options = {'max_to_keep': 1}

    options = ocp.CheckpointManagerOptions(
        best_fn=lambda u: u['loss'],
        best_mode='min',
        step_prefix='ckpt',
        **ckpt_manager_options
    )

    ckpt_mngr = checkpoint_utils.make_checkpoint_manager(
        ckpt_dir=ckpt_dir,
        ckpt_mngr_options=options
    )

    training_step_fn = make_training_step_fn(
        optimizer,
        loss_fn,
        log_gradient_values
    )

    validation_step_fn = make_validation_step_fn(
        val_fn
    )

    processed_graphs = 0
    processed_nodes = 0
    step = 0

    opt_state = None
    for epoch in range(num_epochs):
        # Shuffle the training data.
        numpy_rng.shuffle(training_data)
        # Create batched graphs from list of graphs.
        iterator_training = jraph.dynamically_batch(
            training_data,
            n_node=batch_max_num_nodes,
            n_edge=batch_max_num_edges,
            n_graph=batch_max_num_graphs,
            n_pairs=batch_max_num_pairs,
        )

        # Start iteration over batched graphs.
        for graph_batch_training in iterator_training:
            batch_training = graph_to_batch_fn(graph_batch_training)
            processed_graphs += batch_training['num_of_non_padded_graphs']
            processed_nodes += batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training)
            # Training data is numpy arrays so we now transform them to jax.numpy arrays.
            batch_training = jax.tree_map(jnp.array, batch_training)

            # If params are None (in the first step), initialize the parameters or load from existing checkpoint.
            if params is None:
                # Check if checkpoint already exists.
                latest_step = ckpt_mngr.latest_step()
                if latest_step is not None:
                    if allow_restart:
                        # params = ckpt_mngr.restore(
                        #     latest_step,
                        #     args=ocp.args.Composite(params=ocp.args.StandardRestore())
                        # )['params']
                        params = checkpoint_utils.load_params_from_checkpoint(
                            ckpt_dir=ckpt_dir
                        )
                        step += latest_step
                        print(f'Re-start training from {latest_step}.')
                    else:
                        raise RuntimeError(f'{ckpt_dir} already exists at step {latest_step}. If you want to re-start '
                                           f'training, set `allow_restart=True`.')
                else:
                    params = model.init(jax_rng, batch_training)

            # If optimizer state is None (in the first step), initialize from the parameter pyTree.
            if opt_state is None:
                opt_state = optimizer.init(params)

            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                wandb.log(
                    {f'train_{k}': v for (k, v) in train_metrics_np.items()},
                    step=step
                )

            # Start validation process.
            if step % eval_every_num_steps == 0:
                iterator_validation = jraph.dynamically_batch(
                    validation_data,
                    n_node=batch_max_num_nodes,
                    n_edge=batch_max_num_edges,
                    n_graph=batch_max_num_graphs,
                    n_pairs=batch_max_num_pairs,
                )

                # Start iteration over validation batches.
                eval_metrics: Any = None
                eval_collection: Any = None
                for graph_batch_validation in iterator_validation:
                    batch_validation = graph_to_batch_fn(graph_batch_validation)
                    batch_validation = jax.tree_map(jnp.array, batch_validation)

                    eval_out = validation_step_fn(
                        params,
                        batch_validation
                    )
                    # The metrics are created dynamically during the first evaluation batch, since we aim to support
                    # all kinds of targets beyond energies and forces at some point.
                    if eval_collection is None:
                        eval_collection = clu_metrics.Collection.create(
                            **{k: clu_metrics.Average.from_output(f'{k}') for k in eval_out.keys()})

                    eval_metrics = (
                        eval_collection.single_from_model_output(**eval_out)
                        if eval_metrics is None
                        else eval_metrics.merge(eval_collection.single_from_model_output(**eval_out))
                    )

                eval_metrics = eval_metrics.compute()

                # Convert to dict to log with weights and bias.
                eval_metrics = {
                    f'eval_{k}': float(v) for k, v in eval_metrics.items()
                }

                # Print eval_metrics
                print(print_metrics(f"val_{epoch}_{step}:", eval_metrics))

                # Save checkpoint.
                ckpt_mngr.save(
                    step,
                    args=ocp.args.Composite(params=ocp.args.StandardSave(params)),
                    metrics={
                        'loss': eval_metrics['eval_loss']
                    }
                )

                # Log to weights and bias.
                if use_wandb:
                    wandb.log(
                        eval_metrics,
                        step=step
                    )
            # Finished validation process.

    # Wait until checkpoint manager completes all save operations.
    ckpt_mngr.wait_until_finished()


def fit_from_iterator(
        model,
        optimizer,
        loss_fn,
        graph_to_batch_fn,
        training_iterator,
        validation_iterator,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs,
        batch_max_num_pairs,
        num_epochs,
        num_train,
        num_valid,
        params=None,
        val_fn=None,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart: bool = False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        log_gradient_values: bool = False
):
    """
    Fit model.

    Args:
        model: flax module.
        optimizer: optax optimizer.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        graph_to_batch_fn (Callable): Function that takes a batched graph and returns a batch for the loss_fn.
        training_iterator (): Iterator yielding jraph.GraphTuples.
        validation_iterator (): Iterator yielding jraph.GraphTuples.
        batch_max_num_nodes (int): Maximal number of nodes per batch.
        batch_max_num_edges (int): Maximal number of edges per batch.
        batch_max_num_graphs (int): Maximal number of graphs per batch.
        batch_max_num_pairs (int): Maximal number of pairs in long-range indices.
        num_epochs (int): Number of epochs to train for.
        num_train (int): Number of training examples.
        num_valid (int): Number of validation examples.
        params: Parameters to start from during training. If not given, either new parameters are initialized randomly
            or loaded from ckpt_dir if the checkpoint already exists and `allow_restart=True`.
        val_fn (Callable, optional): Validation function to use for metrics during validation. 
            If None, loss_fn will be used. Defaults to None.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_num_steps (int): Evaluate the metrics every num-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias. If true, wandb.init() must be called before call to fit().
        log_gradient_values (bool): Gradient values for each set of weights is logged.
    Returns:

    """
    del training_seed
    # numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(exist_ok=True)

    # Create orbax CheckpointManager.
    if ckpt_manager_options is None:
        ckpt_manager_options = {'max_to_keep': 1}

    options = ocp.CheckpointManagerOptions(
        best_fn=lambda u: u['loss'],
        best_mode='min',
        step_prefix='ckpt',
        **ckpt_manager_options
    )

    ckpt_mngr = checkpoint_utils.make_checkpoint_manager(
        ckpt_dir=ckpt_dir,
        ckpt_mngr_options=options
    )

    training_step_fn = make_training_step_fn(
        optimizer,
        loss_fn,
        log_gradient_values
    )

    validation_step_fn = make_validation_step_fn(
        val_fn
    )

    processed_graphs = 0
    processed_nodes = 0
    step = 0

    opt_state = None

    for epoch in range(num_epochs):
        if use_wandb:
            wandb.log({"epoch": epoch})

        training_iterator_loop = training_iterator.next_epoch(split=f'train[:{num_train}]', mode = 'train')
        for graph_batch_training in training_iterator_loop:
            batch_training = graph_to_batch_fn(graph_batch_training)
            processed_graphs += batch_training['num_of_non_padded_graphs']
            processed_nodes += batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training)
            # Training data is numpy arrays so we now transform them to jax.numpy arrays.
            batch_training = jax.tree_map(jnp.array, batch_training)

            # If params are None (in the first step), initialize the parameters or load from existing checkpoint.
            if params is None:
                # Check if checkpoint already exists.
                latest_step = ckpt_mngr.latest_step()
                if latest_step is not None:
                    if allow_restart:
                        params = checkpoint_utils.load_params_from_checkpoint(
                            ckpt_dir=ckpt_dir
                        )
                        # params = ckpt_mngr.restore(
                        #     latest_step,
                        #     args=checkpoint.args.Composite(params=checkpoint.args.StandardRestore())
                        # )['params']
                        step += latest_step
                        print(f'Re-start training from {latest_step}.')
                    else:
                        raise RuntimeError(f'{ckpt_dir} already exists at step {latest_step}. If you want to re-start '
                                           f'training, set `allow_restart=True`.')
                else:
                    print(f'Initialize new parameters.')
                    params = model.init(jax_rng, batch_training)

            # If optimizer state is None (in the first step), initialize from the parameter pyTree.
            if opt_state is None:
                opt_state = optimizer.init(params)

            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                wandb.log(
                    {f'train_{k}': v for (k, v) in train_metrics_np.items()},
                    step=step
                )

            # Start validation process.
            if step % eval_every_num_steps == 0:
    #           validation_iterator_batched = jraph.dynamically_batch(
    #               validation_iterator.as_numpy_iterator(),
    #               n_node=batch_max_num_nodes,
    #               n_edge=batch_max_num_edges,
    #               n_graph=batch_max_num_graphs,
    #               n_pairs=batch_max_num_pairs
    #           )

                # Start iteration over validation batches.
                eval_metrics: Any = None
                eval_collection: Any = None
                #for graph_batch_validation in validation_iterator_batched:
                validation_iterator_loop = validation_iterator.next_epoch(split=f'train[-{num_valid}:]', mode = 'validation')
                for graph_batch_validation in validation_iterator_loop:
                    batch_validation = graph_to_batch_fn(graph_batch_validation)
                    batch_validation = jax.tree_map(jnp.array, batch_validation)

                    eval_out = validation_step_fn(
                        params,
                        batch_validation
                    )
                    # The metrics are created dynamically during the first evaluation batch, since we aim to support
                    # all kinds of targets beyond energies and forces at some point.
                    if eval_collection is None:
                        eval_collection = clu_metrics.Collection.create(
                            **{k: clu_metrics.Average.from_output(f'{k}') for k in eval_out.keys()})

                    eval_metrics = (
                        eval_collection.single_from_model_output(**eval_out)
                        if eval_metrics is None
                        else eval_metrics.merge(eval_collection.single_from_model_output(**eval_out))
                    )

                eval_metrics = eval_metrics.compute()

                # Convert to dict to log with weights and bias.
                eval_metrics = {
                    f'eval_{k}': float(v) for k, v in eval_metrics.items()
                }

                print(print_metrics(f"val_{epoch}_{step}:", eval_metrics))
                # Save checkpoint.
                ckpt_mngr.save(
                    step,
                    args=ocp.args.Composite(params=ocp.args.StandardSave(params)),
                    metrics={
                        'loss': eval_metrics['eval_loss']
                    }
                )

                # Log to weights and bias.
                if use_wandb:
                    wandb.log(
                        eval_metrics,
                        step=step
                    )
            # Finished validation process.

    # Wait until checkpoint manager completes all save operations.
    ckpt_mngr.wait_until_finished()


def make_optimizer(
        name: str = 'adam',
        optimizer_args: Dict = dict(),
        learning_rate: float = 1e-3,
        learning_rate_schedule: str = 'constant_schedule',
        learning_rate_schedule_args: Dict = dict(),
        gradient_clipping: str = 'identity',
        gradient_clipping_args: Dict = dict(),
        num_of_nans_to_ignore: int = 0
):
    """Make optax optimizer.

    Args:
        name (str): Name of the optimizer. Defaults to the Adam optimizer.
        optimizer_args (dict): Arguments passed to the optimizer.
        learning_rate (float): Learning rate.
        learning_rate_schedule (str): Learning rate schedule. Defaults to no schedule, meaning learning rate is
            held constant.
        learning_rate_schedule_args (dict): Arguments for the learning rate schedule.
        num_of_nans_to_ignore (int): Number of times NaNs are ignored during in the gradient step. Defaults to 0.
        gradient_clipping (str): Gradient clipping to apply.
        gradient_clipping_args (dict): Arguments to the gradient clipping to apply.
    Returns:

    """
    lr_schedule = getattr(
        optax,
        learning_rate_schedule
    )

    lr_schedule = lr_schedule(
        learning_rate,
        **learning_rate_schedule_args
    )

    opt = getattr(
        optax,
        name
    )

    opt = opt(
        lr_schedule,
        **optimizer_args
    )

    clip_transform = getattr(
        optax,
        gradient_clipping
    )

    clip_transform = clip_transform(
        **gradient_clipping_args
    )

    return optax.chain(
        clip_transform,
        optax.zero_nans(),
        opt
    )


def freeze_parameters(optimizer, trainable_subset_keys):
    """Freeze parameters by giving keys for trainable subsets. Thus, all parameters that are NOT in
    `trainable_subset_keys` are frozen.

    Args:
        optimizer (): optax.GradientTransformation.
        trainable_subset_keys (Sequence): Keys which belong to entries in the PyTree that are trainable. Note that
        for a pyTree like {'a': {'b': *, 'c': *}, 'd': *} and trainable_subset_keys = ['a'] one gets the following
        {'a': {'b': 'trainable', 'c': 'trainable'}, 'd': 'frozen'}. If 'c' and 'd' should be trainable one has to
        pass trainable_subset_keys = ['c', 'd'].

    Returns:

    """

    return optax.multi_transform(
        {'trainable': optimizer, 'frozen': zero_grads()},
        param_labels=make_annotation_fn(trainable_subset_keys)
    )


def make_annotation_fn(keys):
    return lambda params: traverse_util.path_aware_map(
        lambda path, v: 'trainable' if len(set(keys) & set(path)) > 0 else 'frozen', params
    )


def zero_grads():

    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)
