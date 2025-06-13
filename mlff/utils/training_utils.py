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

from .robust_loss_jax import distribution as robust_loss_dist

# Initialize the distribution object for adaptive robust loss
ROBUST_LOSS_DIST = robust_loss_dist.Distribution()

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

    # jax.debug.print("y_label_graph: {}", y_label)
    # jax.debug.print("y_graph: {}", y)
    
    if use_robust_loss:
        # Adaptive robust loss
        diff = jnp.where(full_mask, y - y_label, 0).reshape(-1)

        # # Compute adaptive scale
        # mean_abs_diff = jnp.sum(jnp.abs(diff)) / jnp.maximum(denominator, 1)
        # robust_scale = jnp.maximum(mean_abs_diff * 10, 1e-6)
        # jax.debug.print("robust_scale_graph: {}", robust_scale)
        # jax.debug.print("mean_abs_diff_graph: {}", mean_abs_diff)

        # loss = jnp.sum(2 * scale * ROBUST_LOSS_DIST.nllfun(diff, robust_loss_alpha, robust_scale)) / denominator
        # Compute adaptive scale, use scale of 1.0
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

    # jax.debug.print("y_label_node: {}", y_label)
    # jax.debug.print("y_node: {}", y)

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
        
        # # Compute adaptive scale
        # valid_count = (~jnp.isnan(y_label)).sum()
        # mean_abs_diff = jnp.sum(jnp.abs(masked_diff)) / jnp.maximum(valid_count, 1)
        # robust_scale = jnp.maximum(mean_abs_diff * 10, 1e-6)
        # jax.debug.print("robust_scale_node: {}", robust_scale)
        # jax.debug.print("mean_abs_diff_node: {}", mean_abs_diff)
        
        squared = gradient_utils.safe_mask(
            # fn=lambda u: 2 * ROBUST_LOSS_DIST.nllfun(u, robust_loss_alpha, robust_scale),
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
        loss_mae = jnp.zeros(1)
        metrics = {}
        # Iterate over the targets, calculate loss and multiply with loss weights and scales.
        for target in targets:

            target_mae_fn = property_to_mae[target]
            _mae = target_mae_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask']
            )

            # metrics.update({f'{target}_mae': _mae})
            metrics.update({f'{target}_mae': _mae / _scales[target].mean()})

            loss_mae += weights[target] * _mae

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
        loss_mae = jnp.reshape(loss_mae, ())
        metrics.update({'loss': loss})
        metrics.update({'loss_mae': loss_mae})

        return loss, metrics

    return loss_fn


def make_val_fn(obs_fn: Callable, weights: Dict, scales: Dict = None, 
                use_robust_loss: bool = False, robust_loss_alpha: float = 1.99):
    """Creates a validation function that calculates MAE metrics
    
    Args:
        obs_fn (Callable): Observable function that returns predicted properties
        weights (Dict): Dictionary of property names and their weights
        scales (Dict, optional): Dictionary of scales for each property. Defaults to None.
        use_robust_loss (bool, optional): Whether to use robust loss for MSE calculation. Defaults to False.
        robust_loss_alpha (float, optional): Alpha parameter for robust loss. Defaults to 1.99.
    
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

            # metrics.update({f'{target}_mae': _mae})
            metrics.update({f'{target}_mae': _mae / _scales[target].mean()})

            # Calculate MSE metrics
            target_loss_fn = property_to_loss[target]
            _mse = target_loss_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
                use_robust_loss=use_robust_loss,
                robust_loss_alpha=robust_loss_alpha
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
        log_gradient_values: bool = False,
        use_robust_loss_validation: bool = False,
        robust_loss_alpha_validation: float = 1.99
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
        use_robust_loss_validation (bool): Whether to use robust loss during validation. Defaults to False.
        robust_loss_alpha_validation (float): Alpha parameter for robust loss during validation. Defaults to 1.99.
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


    # Print all parameter keys and shapes
    def print_param_shapes(params, prefix=''):
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (dict, jnp.ndarray)):
                    if isinstance(value, jnp.ndarray):
                        print(f"{prefix}{key}: {value.shape}")
                    else:
                        print(f"{prefix}{key}:")
                        print_param_shapes(value, prefix + '  ')


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
                        print(f"Loaded parameters from {ckpt_dir}")
                        print(f"Params keys: {params.keys()}")
                        print("\nParameter shapes:")
                        print("=" * 50)
                        print_param_shapes(params)
                        print("=" * 50)
                        print('This is fit_from_iterator function')
                        # Modify parameters to handle theory levels
                        if 'params' in params and 'observables_0' in params['params']:
                            num_theory_levels = 16
                            # Modify energy_offset
                            if 'energy_offset' in params['params']['observables_0']:
                                print("\nOriginal energy_offset:")
                                print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                print("Values:", params['params']['observables_0']['energy_offset'])
                                old_energy_offset = params['params']['observables_0']['energy_offset']
                                
                                # Only tile if shape is 1D
                                if len(old_energy_offset.shape) == 1:
                                    new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
                                    params['params']['observables_0']['energy_offset'] = new_energy_offset
                                    print("Applied tiling to energy_offset")
                                else:
                                    print("Energy offset already has multiple dimensions, no tiling applied")
                                
                                print("\nNew energy_offset:")
                                print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                print("Values:", params['params']['observables_0']['energy_offset'])

                            # Modify atomic_scales
                            if 'atomic_scales' in params['params']['observables_0']:
                                print("\nOriginal atomic_scales:")
                                print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                print("Values:", params['params']['observables_0']['atomic_scales'])
                                old_atomic_scales = params['params']['observables_0']['atomic_scales']
                                
                                # Only tile if shape is 1D
                                if len(old_atomic_scales.shape) == 1:
                                    new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
                                    params['params']['observables_0']['atomic_scales'] = new_atomic_scales
                                    print("Applied tiling to atomic_scales")
                                else:
                                    print("Atomic scales already has multiple dimensions, no tiling applied")
                                
                                print("\nNew atomic_scales:")
                                print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                print("Values:", params['params']['observables_0']['atomic_scales'])

                            # Modify energy_dense_final
                            if 'energy_dense_final' in params['params']['observables_0']:
                                print("\nOriginal energy_dense_final kernel:")
                                print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])
                                old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']
                                
                                # Check the shape to determine if tiling is needed
                                if old_kernel.shape[1] == 1:
                                    new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
                                    params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel
                                    print("Applied tiling to energy_dense_final kernel")
                                else:
                                    print("Energy dense final kernel already has correct output dimension, no tiling applied")
                                
                                print("\nNew energy_dense_final kernel:")
                                print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])

                            print("\nParameter shapes after modification:")
                            print("=" * 50)
                            print_param_shapes(params)
                            print("=" * 50)
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
        params=None,
        val_fn=None,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart: bool = False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        log_gradient_values: bool = False,
        use_robust_loss_validation: bool = False,
        robust_loss_alpha_validation: float = 1.99
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
        use_robust_loss_validation (bool): Whether to use robust loss during validation. Defaults to False.
        robust_loss_alpha_validation (float): Alpha parameter for robust loss during validation. Defaults to 1.99.
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

    # Print all parameter keys and shapes
    def print_param_shapes(params, prefix=''):
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (dict, jnp.ndarray)):
                    if isinstance(value, jnp.ndarray):
                        print(f"{prefix}{key}: {value.shape}")
                    else:
                        print(f"{prefix}{key}:")
                        print_param_shapes(value, prefix + '  ')

    processed_graphs = 0
    processed_nodes = 0
    step = 0

    opt_state = None

    for epoch in range(num_epochs):
        if use_wandb:
            wandb.log({"epoch": epoch})
        print(f'Epoch {epoch} of {num_epochs}')
        training_iterator_loop = training_iterator.next_epoch(split='train', mode='train')
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
                        print(f"Loaded parameters from {ckpt_dir}")
                        print(f"Params keys: {params.keys()}")
                        print("\nParameter shapes:")
                        print("=" * 50)
                        print_param_shapes(params)
                        print("=" * 50)
                        print('This is fit_from_iterator function')
                        # Modify parameters to handle theory levels
                        if 'params' in params and 'observables_0' in params['params']:
                            num_theory_levels = 16
                            # Modify energy_offset
                            if 'energy_offset' in params['params']['observables_0']:
                                print("\nOriginal energy_offset:")
                                print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                print("Values:", params['params']['observables_0']['energy_offset'])
                                old_energy_offset = params['params']['observables_0']['energy_offset']
                                
                                # Only tile if shape is 1D
                                if len(old_energy_offset.shape) == 1:
                                    new_energy_offset = jnp.tile(old_energy_offset[:, None], (1, num_theory_levels))
                                    params['params']['observables_0']['energy_offset'] = new_energy_offset
                                    print("Applied tiling to energy_offset")
                                else:
                                    print("Energy offset already has multiple dimensions, no tiling applied")
                                
                                print("\nNew energy_offset:")
                                print("Shape:", params['params']['observables_0']['energy_offset'].shape)
                                print("Values:", params['params']['observables_0']['energy_offset'])

                            # Modify atomic_scales
                            if 'atomic_scales' in params['params']['observables_0']:
                                print("\nOriginal atomic_scales:")
                                print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                print("Values:", params['params']['observables_0']['atomic_scales'])
                                old_atomic_scales = params['params']['observables_0']['atomic_scales']
                                
                                # Only tile if shape is 1D
                                if len(old_atomic_scales.shape) == 1:
                                    new_atomic_scales = jnp.tile(old_atomic_scales[:, None], (1, num_theory_levels))
                                    params['params']['observables_0']['atomic_scales'] = new_atomic_scales
                                    print("Applied tiling to atomic_scales")
                                else:
                                    print("Atomic scales already has multiple dimensions, no tiling applied")
                                
                                print("\nNew atomic_scales:")
                                print("Shape:", params['params']['observables_0']['atomic_scales'].shape)
                                print("Values:", params['params']['observables_0']['atomic_scales'])

                            # Modify energy_dense_final
                            if 'energy_dense_final' in params['params']['observables_0']:
                                print("\nOriginal energy_dense_final kernel:")
                                print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])
                                old_kernel = params['params']['observables_0']['energy_dense_final']['kernel']
                                
                                # Check the shape to determine if tiling is needed
                                if old_kernel.shape[1] == 1:
                                    new_kernel = jnp.tile(old_kernel, (1, num_theory_levels))
                                    params['params']['observables_0']['energy_dense_final']['kernel'] = new_kernel
                                    print("Applied tiling to energy_dense_final kernel")
                                else:
                                    print("Energy dense final kernel already has correct output dimension, no tiling applied")
                                
                                print("\nNew energy_dense_final kernel:")
                                print("Shape:", params['params']['observables_0']['energy_dense_final']['kernel'].shape)
                                print("Values:", params['params']['observables_0']['energy_dense_final']['kernel'])

                            print("\nParameter shapes after modification:")
                            print("=" * 50)
                            print_param_shapes(params)
                            print("=" * 50)

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
                # Start iteration over validation batches.
                eval_metrics: Any = None
                eval_collection: Any = None
                validation_iterator_loop = validation_iterator.next_epoch(split='train', mode='validation')
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
