import jax
import jax.numpy as jnp
import jraph


def batch_info_fn(batched_graph: jraph.GraphsTuple):
    """
    Collect batching information from batched `jraph.GraphsTuple`.
    Args:
        batched_graph (jraph.GraphsTuple): Batched `jraph.GraphsTuple`

    Returns: Dictionary with `node_mask`, `graph_mask` and `batch_segments`.

    Raises:
        RuntimeError: If a non-batched `jraph.GraphsTuple` is passed as input.

    """
    node_mask = jraph.get_node_padding_mask(batched_graph)
    graph_mask = jraph.get_graph_padding_mask(batched_graph)

    num_of_non_padded_graphs = len(graph_mask) - jraph.get_number_of_padding_with_graphs_graphs(batched_graph)
    if len(graph_mask) == 1:
        raise RuntimeError('Only batched `jraph.GraphsTuple` should be passed to `batch_info_fn`.')
    
    batch_segments = jnp.repeat(
        jnp.arange(len(graph_mask)),
        repeats=batched_graph.n_node,
        total_repeat_length=len(node_mask)
    )

    # Get theory_level from globals (shape: (num_graphs,))
    theory_level = batched_graph.globals.get('theory_level')
    if theory_level is not None:
        # Create one-hot encoded mask for each graph's theory level
        # We'll use a fixed maximum number of theory levels (e.g., 3 for high, medium, low)
        max_theory_levels = 3  # Fixed number of theory levels
        graph_theory_mask = jnp.zeros((len(graph_mask), max_theory_levels), dtype=jnp.bool_)
        # Only set True for valid theory levels (0, 1, 2)
        valid_levels = (theory_level >= 0) & (theory_level < max_theory_levels)
        graph_theory_mask = graph_theory_mask.at[jnp.arange(len(graph_mask)), theory_level].set(valid_levels)
        
        # Broadcast to node level
        # shape: (num_nodes, num_theory_levels)
        theory_mask = jnp.repeat(
            graph_theory_mask,
            repeats=batched_graph.n_node,
            axis=0,
            total_repeat_length=len(node_mask)
        )
    else:
        # Default to single theory level if not provided
        theory_mask = jnp.ones((len(node_mask), 1), dtype=jnp.bool_)

    return dict(
        node_mask=node_mask,
        graph_mask=graph_mask,
        batch_segments=batch_segments,
        num_of_non_padded_graphs=num_of_non_padded_graphs,
        theory_mask=theory_mask,
    )


@jax.jit
def graph_to_batch_fn(graph: jraph.GraphsTuple):
    batch = dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        num_unpaired_electrons=graph.globals.get('num_unpaired_electrons'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=graph.edges.get('cell'),
        cell_offset=graph.edges.get('cell_offset'),
        energy=graph.globals.get('energy'),
        forces=graph.nodes.get('forces'),
        stress=graph.globals.get('stress'),
        total_charge=graph.globals.get('total_charge'),
        dipole_vec=graph.globals.get('dipole_vec'),
        hirshfeld_ratios=graph.nodes.get('hirshfeld_ratios'),
        idx_i_lr=graph.idx_i_lr,
        idx_j_lr=graph.idx_j_lr,
        theory_level=graph.globals.get('theory_level'),
    )
    batch_info = batch_info_fn(graph)
    batch.update(batch_info)
    return batch


def graph_to_inputs_fn(graph: jraph.GraphsTuple):
    """
    Wrapper around graph_to_batch_fn.

    Args:
        graph ():

    Returns:
        Dictionary representation of the graph.

    """
    return graph_to_batch_fn(graph)
