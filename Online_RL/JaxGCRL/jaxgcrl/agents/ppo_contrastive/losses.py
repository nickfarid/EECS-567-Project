from typing import Any, Tuple

import jax
import jax.numpy as jnp
from brax.training import types

from .networks import PPOContrastiveNetworkParams, PPOContrastiveNetworks


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
):
    """Calculates the Generalized Advantage Estimation (GAE)."""

    truncation_mask = 1 - truncation
    values_t_plus_1 = jnp.concatenate([values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask
    acc = jnp.zeros_like(bootstrap_value)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), acc

    (_, _), vs_minus_v_xs = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    vs = jnp.add(vs_minus_v_xs, values)
    vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount * (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def _l2_normalize(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + 1e-8)


def _normalize_goal(
    ppo_network: PPOContrastiveNetworks,
    normalizer_params: Any,
    goal: jnp.ndarray,
) -> jnp.ndarray:
    state_padding = jnp.zeros(goal.shape[:-1] + (ppo_network.state_dim,), dtype=goal.dtype)
    goal_obs = jnp.concatenate([state_padding, goal], axis=-1)
    normalized_goal_obs = ppo_network.preprocess_observations_fn(goal_obs, normalizer_params)
    return normalized_goal_obs[..., ppo_network.state_dim :]


def compute_contrastive_loss(
    params: PPOContrastiveNetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    ppo_network: PPOContrastiveNetworks,
    tau: float,
) -> Tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    state = ppo_network.preprocess_observations_fn(data.observation, normalizer_params)[
        ..., : ppo_network.state_dim
    ]
    future_goal = data.extras["contrastive_extras"]["future_goal"]
    valid_mask = data.extras["contrastive_extras"]["valid_mask"].astype(jnp.float32)
    normalized_future_goal = _normalize_goal(ppo_network, normalizer_params, future_goal)

    state_repr = ppo_network.state_encoder_network.apply(
        None, params.encoder["state"], state.reshape((-1, state.shape[-1]))
    )
    goal_repr = ppo_network.goal_encoder_network.apply(
        None, params.encoder["goal"], normalized_future_goal.reshape((-1, normalized_future_goal.shape[-1]))
    )
    state_repr = _l2_normalize(state_repr)
    goal_repr = _l2_normalize(goal_repr)

    valid_mask = valid_mask.reshape((-1,))
    logits = jnp.matmul(state_repr, goal_repr.T) / tau

    pair_mask = jnp.outer(valid_mask, valid_mask).astype(bool)
    identity = jnp.eye(logits.shape[0], dtype=bool)
    safe_mask = pair_mask | identity
    masked_logits = jnp.where(safe_mask, logits, -1e9)

    log_partition = jax.nn.logsumexp(masked_logits, axis=1)
    positive_logits = jnp.diag(masked_logits)
    valid_pairs = jnp.maximum(jnp.sum(valid_mask), 1.0)
    contrastive_loss = -jnp.sum(valid_mask * (positive_logits - log_partition)) / valid_pairs

    predictions = jnp.argmax(masked_logits, axis=1)
    contrastive_accuracy = jnp.sum(valid_mask * (predictions == jnp.arange(logits.shape[0]))) / valid_pairs
    contrastive_loss = jnp.where(jnp.sum(valid_mask) > 0, contrastive_loss, 0.0)
    contrastive_accuracy = jnp.where(jnp.sum(valid_mask) > 0, contrastive_accuracy, 0.0)

    metrics = {
        "contrastive_loss": contrastive_loss,
        "contrastive_pairs": jnp.sum(valid_mask),
        "contrastive_accuracy": contrastive_accuracy,
    }
    return contrastive_loss, metrics


def compute_ppo_contrastive_loss(
    params: PPOContrastiveNetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: PPOContrastiveNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    lambda_con: float = 0.1,
    tau: float = 0.1,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes PPO loss with an auxiliary contrastive term."""

    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    policy_logits = policy_apply(normalizer_params, params, data.observation)
    baseline = value_apply(normalizer_params, params, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params, terminal_obs)

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)
    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    gae_returns = jax.lax.stop_gradient(jnp.add(advantages, jax.lax.stop_gradient(baseline)))

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    ppo_loss = policy_loss + v_loss + entropy_loss
    contrastive_loss, contrastive_metrics = compute_contrastive_loss(
        params=params,
        normalizer_params=normalizer_params,
        data=data,
        ppo_network=ppo_network,
        tau=tau,
    )
    total_loss = ppo_loss + lambda_con * contrastive_loss

    new_dist = parametric_action_distribution.create_dist(policy_logits)
    if hasattr(new_dist, "kl_divergence"):
        old_dist_params = data.extras["policy_extras"]["distribution_params"]
        old_dist = parametric_action_distribution.create_dist(old_dist_params)
        kl = jnp.mean(new_dist.kl_divergence(old_dist))
    else:
        kl = jnp.array(0.0)

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "kl_mean": kl,
        **contrastive_metrics,
    }
