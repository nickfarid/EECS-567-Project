from typing import Any, Callable, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class PPOContrastiveNetworkParams:
    encoder: types.Params
    policy: types.Params
    value: types.Params


@flax.struct.dataclass
class PPOContrastiveNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    state_encoder_network: networks.FeedForwardNetwork
    goal_encoder_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    preprocess_observations_fn: types.PreprocessObservationFn
    state_dim: int


class MLP(linen.Module):
    """Simple MLP used for encoders and heads."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.swish
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


def _make_mlp_network(
    input_size: int,
    output_size: int,
    hidden_layer_sizes: Sequence[int],
    activation: ActivationFn,
    activate_final: bool = False,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    module = MLP(
        layer_sizes=tuple(hidden_layer_sizes) + (output_size,),
        activation=activation,
        activate_final=activate_final,
        layer_norm=layer_norm,
    )
    dummy_input = jnp.zeros((1, input_size))
    return networks.FeedForwardNetwork(
        init=lambda key: module.init(key, dummy_input),
        apply=lambda _, params, data: module.apply(params, data),
    )


def _encode_obs(
    ppo_networks: PPOContrastiveNetworks,
    params: PPOContrastiveNetworkParams,
    normalizer_params: Any,
    obs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    normalized_obs = ppo_networks.preprocess_observations_fn(obs, normalizer_params)
    state = normalized_obs[..., : ppo_networks.state_dim]
    goal = normalized_obs[..., ppo_networks.state_dim :]
    state_latent = ppo_networks.state_encoder_network.apply(None, params.encoder["state"], state)
    goal_latent = ppo_networks.goal_encoder_network.apply(None, params.encoder["goal"], goal)
    return state_latent, goal_latent


def make_inference_fn(ppo_networks: PPOContrastiveNetworks):
    """Creates params and inference function for the PPO contrastive agent."""

    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
        def policy(observations: types.Observation, key_sample: PRNGKey):
            logits = ppo_networks.policy_network.apply(*params, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}

            raw_actions = ppo_networks.parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = ppo_networks.parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = ppo_networks.parametric_action_distribution.postprocess(raw_actions)
            extras = {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
            }
            return postprocessed_actions, extras

        return policy

    return make_policy


def make_ppo_contrastive_networks(
    observation_size: int,
    action_size: int,
    state_dim: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (256, 256),
    latent_dim: int = 64,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: ActivationFn = linen.swish,
) -> PPOContrastiveNetworks:
    goal_dim = observation_size - state_dim
    feature_dim = 2 * latent_dim
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    state_encoder_network = _make_mlp_network(
        input_size=state_dim,
        output_size=latent_dim,
        hidden_layer_sizes=encoder_hidden_layer_sizes,
        activation=activation,
    )
    goal_encoder_network = _make_mlp_network(
        input_size=goal_dim,
        output_size=latent_dim,
        hidden_layer_sizes=encoder_hidden_layer_sizes,
        activation=activation,
    )
    policy_head_network = _make_mlp_network(
        input_size=feature_dim,
        output_size=parametric_action_distribution.param_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
    )
    value_head_network = _make_mlp_network(
        input_size=feature_dim,
        output_size=1,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
    )

    networks_bundle = PPOContrastiveNetworks(
        policy_network=None,  # type: ignore[arg-type]
        value_network=None,  # type: ignore[arg-type]
        state_encoder_network=state_encoder_network,
        goal_encoder_network=goal_encoder_network,
        parametric_action_distribution=parametric_action_distribution,
        preprocess_observations_fn=preprocess_observations_fn,
        state_dim=state_dim,
    )

    def policy_apply(normalizer_params: Any, params: PPOContrastiveNetworkParams, obs: jnp.ndarray):
        state_latent, goal_latent = _encode_obs(networks_bundle, params, normalizer_params, obs)
        features = jnp.concatenate([state_latent, goal_latent], axis=-1)
        return policy_head_network.apply(None, params.policy, features)

    def value_apply(normalizer_params: Any, params: PPOContrastiveNetworkParams, obs: jnp.ndarray):
        state_latent, goal_latent = _encode_obs(networks_bundle, params, normalizer_params, obs)
        features = jnp.concatenate([state_latent, goal_latent], axis=-1)
        return jnp.squeeze(value_head_network.apply(None, params.value, features), axis=-1)

    return networks_bundle.replace(
        policy_network=networks.FeedForwardNetwork(
            init=policy_head_network.init,
            apply=policy_apply,
        ),
        value_network=networks.FeedForwardNetwork(
            init=value_head_network.init,
            apply=value_apply,
        ),
    )
