from typing import Tuple

import jax
import jax.numpy as jnp


def sample_future_goals(
    observation: jnp.ndarray,
    traj_id: jnp.ndarray,
    state_dim: int,
    goal_indices: Tuple[int, ...],
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Samples one future achieved goal per timestep from the same trajectory.

    Args:
        observation: [num_envs, total_steps, obs_dim] array.
        traj_id: [num_envs, total_steps] trajectory ids aligned with observation.
        state_dim: Size of the state component in observation.
        goal_indices: Indices of goal coordinates inside the state component.
        key: Random key used for categorical sampling.
    """

    total_steps = observation.shape[1]
    time_index = jnp.arange(total_steps)
    future_mask = time_index[None, :, None] < time_index[None, None, :]
    same_traj = traj_id[:, :, None] == traj_id[:, None, :]
    valid_future = future_mask & same_traj
    valid_rows = jnp.any(valid_future, axis=-1)

    fallback_mask = jax.nn.one_hot(
        jnp.zeros_like(traj_id, dtype=jnp.int32), total_steps, dtype=bool
    )
    safe_mask = jnp.where(valid_rows[..., None], valid_future, fallback_mask)
    logits = jnp.where(safe_mask, 0.0, -1e9)

    flat_logits = logits.reshape((-1, total_steps))
    flat_keys = jax.random.split(key, flat_logits.shape[0])
    flat_future_index = jax.vmap(jax.random.categorical)(flat_keys, flat_logits)
    future_index = flat_future_index.reshape(traj_id.shape)

    gather_future_obs = jax.vmap(lambda obs, idx: jnp.take(obs, idx, axis=0), in_axes=(0, 0))
    future_obs = gather_future_obs(observation, future_index)
    future_state = future_obs[..., :state_dim]
    future_goal = future_state[..., jnp.array(goal_indices)]
    return future_goal, valid_rows


def build_rollout_contrastive_targets(
    observation: jnp.ndarray,
    traj_id: jnp.ndarray,
    state_dim: int,
    goal_indices: Tuple[int, ...],
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Builds future achieved goals aligned with PPO's [B, T] rollout layout.

    Args:
        observation: [rollout_chunks, unroll_length, num_envs, obs_dim] rollout batch.
        traj_id: [rollout_chunks, unroll_length, num_envs] trajectory ids.
    """

    rollout_chunks, unroll_length, num_envs = observation.shape[:3]
    per_env_obs = jnp.transpose(observation, (2, 0, 1, 3)).reshape((num_envs, -1, observation.shape[-1]))
    per_env_traj_id = jnp.transpose(traj_id, (2, 0, 1)).reshape((num_envs, -1))
    future_goal, valid_mask = sample_future_goals(
        per_env_obs, per_env_traj_id, state_dim, goal_indices, key
    )

    future_goal = future_goal.reshape((num_envs, rollout_chunks, unroll_length, -1))
    future_goal = jnp.transpose(future_goal, (1, 0, 2, 3)).reshape(
        (rollout_chunks * num_envs, unroll_length, -1)
    )
    valid_mask = valid_mask.reshape((num_envs, rollout_chunks, unroll_length))
    valid_mask = jnp.transpose(valid_mask, (1, 0, 2)).reshape((rollout_chunks * num_envs, unroll_length))
    return future_goal, valid_mask
