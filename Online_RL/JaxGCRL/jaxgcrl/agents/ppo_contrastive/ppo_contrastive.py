import functools
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import acting, gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
from orbax import checkpoint as ocp

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator

from .losses import compute_ppo_contrastive_loss
from .networks import (
    PPOContrastiveNetworkParams,
    make_inference_fn,
    make_ppo_contrastive_networks,
)
from .sampling import build_rollout_contrastive_targets

InferenceParams = Tuple[running_statistics.NestedMeanStd, PPOContrastiveNetworkParams]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: PPOContrastiveNetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _flatten_rollout(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.swapaxes(x, 1, 2)
    return jnp.reshape(x, (-1,) + x.shape[2:])


@dataclass
class PPOContrastive:
    """PPO baseline with dual encoders and an auxiliary contrastive loss."""

    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    discounting: float = 0.9
    unroll_length: int = 10
    batch_size: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 2
    num_resets_per_eval: int = 0
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    clipping_epsilon: float = 0.3
    gae_lambda: float = 0.95
    deterministic_eval: bool = False
    normalize_advantage: bool = True
    restore_checkpoint_path: Optional[str] = None
    train_step_multiplier = 1

    encoder_hidden_dim: int = 256
    latent_dim: int = 64
    lambda_con: float = 0.1
    tau: float = 0.1

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        assert self.batch_size * self.num_minibatches % config.num_envs == 0
        state_dim = train_env.state_dim
        goal_indices = tuple(int(idx) for idx in np.asarray(train_env.goal_indices))
        xt = time.time()

        process_count = jax.process_count()
        process_id = jax.process_index()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count
        if config.max_devices_per_host:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)

        logging.info(
            "Device count: %d, process count: %d (id %d), local device count: %d, devices to be used count: %d",
            jax.device_count(),
            process_count,
            process_id,
            local_device_count,
            local_devices_to_use,
        )
        device_count = local_devices_to_use * process_count

        utd_ratio = self.batch_size * self.unroll_length * self.num_minibatches * config.action_repeat
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_training_steps_per_epoch = np.ceil(
            config.total_env_steps / (num_evals_after_init * utd_ratio * max(self.num_resets_per_eval, 1))
        ).astype(int)

        key = jax.random.PRNGKey(config.seed)
        global_key, local_key = jax.random.split(key)
        del key
        local_key = jax.random.fold_in(local_key, process_id)
        local_key, key_env, eval_key = jax.random.split(local_key, 3)
        key_encoder, key_policy, key_value = jax.random.split(global_key, 3)
        del global_key

        assert config.num_envs % device_count == 0

        v_randomization_fn = None
        if randomization_fn is not None:
            randomization_batch_size = config.num_envs // local_device_count
            randomization_rng = jax.random.split(key_env, randomization_batch_size)
            v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

        if isinstance(train_env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        env = TrajectoryIdWrapper(train_env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        reset_fn = jax.jit(jax.vmap(env.reset))
        key_envs = jax.random.split(key_env, config.num_envs // process_count)
        key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
        env_state = reset_fn(key_envs)

        normalize = lambda obs, params: obs
        if self.normalize_observations:
            normalize = running_statistics.normalize

        ppo_network = make_ppo_contrastive_networks(
            env_state.obs.shape[-1],
            env.action_size,
            state_dim=state_dim,
            preprocess_observations_fn=normalize,
            encoder_hidden_layer_sizes=(self.encoder_hidden_dim, self.encoder_hidden_dim),
            latent_dim=self.latent_dim,
        )
        make_policy = make_inference_fn(ppo_network)
        optimizer = optax.adam(learning_rate=self.learning_rate)

        loss_fn = functools.partial(
            compute_ppo_contrastive_loss,
            ppo_network=ppo_network,
            entropy_cost=self.entropy_cost,
            discounting=self.discounting,
            reward_scaling=self.reward_scaling,
            gae_lambda=self.gae_lambda,
            clipping_epsilon=self.clipping_epsilon,
            normalize_advantage=self.normalize_advantage,
            lambda_con=self.lambda_con,
            tau=self.tau,
        )

        gradient_update_fn = gradients.gradient_update_fn(
            loss_fn,
            optimizer,
            pmap_axis_name=_PMAP_AXIS_NAME,
            has_aux=True,
        )

        def minibatch_step(
            carry,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, params, key = carry
            key, key_loss = jax.random.split(key)
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state=optimizer_state,
            )

            return (optimizer_state, params, key), metrics

        def update_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, params, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (self.num_minibatches, -1) + x.shape[1:])
                return x

            shuffled_data = jax.tree_util.tree_map(convert_data, data)
            (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(minibatch_step, normalizer_params=normalizer_params),
                (optimizer_state, params, key_grad),
                shuffled_data,
                length=self.num_minibatches,
            )
            return (optimizer_state, params, key), metrics

        def training_step(
            carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
        ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
            training_state, state, key = carry
            update_key, contrastive_key, key_generate_unroll, new_key = jax.random.split(key, 4)

            policy = make_policy(
                (
                    training_state.normalizer_params,
                    training_state.params,
                )
            )

            def f(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = acting.generate_unroll(
                    env,
                    current_state,
                    policy,
                    current_key,
                    self.unroll_length,
                    extra_fields=("truncation", "traj_id"),
                )
                return (next_state, next_key), data

            (state, _), raw_data = jax.lax.scan(
                f,
                (state, key_generate_unroll),
                (),
                length=self.batch_size * self.num_minibatches // config.num_envs,
            )

            future_goal, valid_mask = build_rollout_contrastive_targets(
                raw_data.observation,
                raw_data.extras["state_extras"]["traj_id"],
                state_dim=state_dim,
                goal_indices=goal_indices,
                key=contrastive_key,
            )
            data = jax.tree_util.tree_map(_flatten_rollout, raw_data)
            data = data._replace(
                extras={
                    **data.extras,
                    "contrastive_extras": {
                        "future_goal": future_goal,
                        "valid_mask": valid_mask.astype(jnp.float32),
                    },
                }
            )
            assert data.discount.shape[1:] == (self.unroll_length,)

            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

            (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(
                    update_step,
                    data=data,
                    normalizer_params=normalizer_params,
                ),
                (
                    training_state.optimizer_state,
                    training_state.params,
                    update_key,
                ),
                (),
                length=self.num_updates_per_batch,
            )

            new_training_state = TrainingState(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + utd_ratio,
            )
            return (new_training_state, state, new_key), metrics

        def training_epoch(
            training_state: TrainingState,
            state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step,
                (training_state, state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
            return training_state, state, loss_metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            nonlocal training_walltime
            t = time.time()
            training_state, env_state = _strip_weak_type((training_state, env_state))
            result = training_epoch(training_state, env_state, key)
            training_state, env_state, metrics = _strip_weak_type(result)

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                num_training_steps_per_epoch * utd_ratio * max(self.num_resets_per_eval, 1)
            ) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            return training_state, env_state, metrics

        encoder_params = {
            "state": ppo_network.state_encoder_network.init(key_encoder),
            "goal": ppo_network.goal_encoder_network.init(jax.random.fold_in(key_encoder, 1)),
        }
        init_params = PPOContrastiveNetworkParams(
            encoder=encoder_params,
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )

        training_state = TrainingState(
            optimizer_state=optimizer.init(init_params),
            params=init_params,
            normalizer_params=running_statistics.init_state(
                specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
            ),
            env_steps=0,
        )

        if config.total_env_steps == 0:
            return make_policy, (training_state.normalizer_params, training_state.params), {}

        if self.restore_checkpoint_path is not None and epath.Path(self.restore_checkpoint_path).exists():
            logging.info("restoring from checkpoint %s", self.restore_checkpoint_path)
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            target = training_state.normalizer_params, init_params
            (normalizer_params, init_params) = orbax_checkpointer.restore(
                self.restore_checkpoint_path, item=target
            )
            training_state = training_state.replace(normalizer_params=normalizer_params, params=init_params)

        training_state = jax.device_put_replicated(
            training_state,
            jax.local_devices()[:local_devices_to_use],
        )

        if not eval_env:
            eval_env = train_env
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(eval_key, config.num_eval_envs),
            )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = wrap_for_training(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        evaluator = Evaluator(
            eval_env,
            functools.partial(
                make_policy,
                deterministic=self.deterministic_eval,
            ),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.params)),
                training_metrics={},
            )
            progress_fn(
                0,
                metrics,
                make_policy,
                _unpmap((training_state.normalizer_params, training_state.params)),
                unwrapped_env,
            )

        training_metrics = {}
        training_walltime = 0
        current_step = 0
        for eval_epoch_num in range(num_evals_after_init):
            logging.info("starting iteration %s %s", eval_epoch_num, time.time() - xt)

            for _ in range(max(self.num_resets_per_eval, 1)):
                epoch_key, local_key = jax.random.split(local_key)
                epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
                training_state, env_state, training_metrics = training_epoch_with_timing(
                    training_state,
                    env_state,
                    epoch_keys,
                )
                current_step = int(_unpmap(training_state.env_steps))

                key_envs = jax.vmap(lambda x, s: jax.random.split(x[0], s), in_axes=(0, None))(
                    key_envs, key_envs.shape[1]
                )
                env_state = reset_fn(key_envs) if self.num_resets_per_eval > 0 else env_state

                if process_id == 0:
                    metrics = evaluator.run_evaluation(
                        _unpmap((training_state.normalizer_params, training_state.params)),
                        training_metrics,
                    )
                    do_render = (eval_epoch_num % config.visualization_interval) == 0
                    progress_fn(
                        current_step,
                        metrics,
                        make_policy,
                        _unpmap((training_state.normalizer_params, training_state.params)),
                        unwrapped_env,
                        do_render=do_render,
                    )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        pmap.assert_is_replicated(training_state)
        params = _unpmap((training_state.normalizer_params, training_state.params))
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
