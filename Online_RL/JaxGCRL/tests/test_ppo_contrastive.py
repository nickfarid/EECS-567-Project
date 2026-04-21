from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxgcrl.agents.ppo_contrastive.losses import compute_ppo_contrastive_loss
from jaxgcrl.agents.ppo_contrastive.networks import (
    PPOContrastiveNetworkParams,
    make_ppo_contrastive_networks,
)
from jaxgcrl.agents.ppo_contrastive.sampling import sample_future_goals


class DummyTransition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: dict


def test_sample_future_goals_respects_traj_boundaries():
    observation = jnp.array(
        [
            [
                [0.0, 10.0, 0.0, 0.0, 99.0, 98.0],
                [1.0, 11.0, 0.0, 0.0, 99.0, 98.0],
                [2.0, 12.0, 0.0, 0.0, 99.0, 98.0],
                [3.0, 13.0, 0.0, 0.0, 99.0, 98.0],
            ]
        ]
    )
    traj_id = jnp.array([[0.0, 0.0, 1.0, 1.0]])
    future_goal, valid_mask = sample_future_goals(
        observation=observation,
        traj_id=traj_id,
        state_dim=4,
        goal_indices=(0, 1),
        key=jax.random.PRNGKey(0),
    )

    assert valid_mask.tolist() == [[True, False, True, False]]
    assert jnp.allclose(future_goal[0, 0], jnp.array([1.0, 11.0]))
    assert jnp.allclose(future_goal[0, 2], jnp.array([3.0, 13.0]))


def test_ppo_contrastive_network_and_loss_shapes():
    observation_size = 6
    state_dim = 4
    action_size = 3
    ppo_network = make_ppo_contrastive_networks(
        observation_size=observation_size,
        action_size=action_size,
        state_dim=state_dim,
        preprocess_observations_fn=lambda obs, _: obs,
    )

    encoder_key, policy_key, value_key, sample_key = jax.random.split(jax.random.PRNGKey(0), 4)
    params = PPOContrastiveNetworkParams(
        encoder={
            "state": ppo_network.state_encoder_network.init(encoder_key),
            "goal": ppo_network.goal_encoder_network.init(jax.random.fold_in(encoder_key, 1)),
        },
        policy=ppo_network.policy_network.init(policy_key),
        value=ppo_network.value_network.init(value_key),
    )

    observation = jnp.arange(2 * 3 * observation_size, dtype=jnp.float32).reshape(2, 3, observation_size)
    policy_logits = ppo_network.policy_network.apply(None, params, observation)
    values = ppo_network.value_network.apply(None, params, observation)

    assert policy_logits.shape[:2] == (2, 3)
    assert values.shape == (2, 3)

    raw_action = jnp.zeros((2, 3, action_size), dtype=jnp.float32)
    behaviour_log_prob = ppo_network.parametric_action_distribution.log_prob(policy_logits, raw_action)
    future_goal = observation[..., :state_dim][..., (0, 1)]

    loss, metrics = compute_ppo_contrastive_loss(
        params=params,
        normalizer_params=None,
        data=DummyTransition(
            observation=observation,
            action=jnp.zeros((2, 3, action_size), dtype=jnp.float32),
            reward=jnp.ones((2, 3), dtype=jnp.float32),
            discount=jnp.ones((2, 3), dtype=jnp.float32) * 0.99,
            next_observation=observation + 1.0,
            extras={
                "policy_extras": {
                    "raw_action": raw_action,
                    "log_prob": behaviour_log_prob,
                    "distribution_params": policy_logits,
                },
                "state_extras": {
                    "truncation": jnp.zeros((2, 3), dtype=jnp.float32),
                },
                "contrastive_extras": {
                    "future_goal": future_goal,
                    "valid_mask": jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=jnp.float32),
                },
            },
        ),
        rng=sample_key,
        ppo_network=ppo_network,
    )

    assert loss.shape == ()
    assert metrics["total_loss"].shape == ()
    assert metrics["contrastive_loss"].shape == ()
