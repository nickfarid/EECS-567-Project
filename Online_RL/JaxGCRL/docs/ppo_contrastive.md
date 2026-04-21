# PPO Contrastive Usage

This repo now includes a new baseline named `ppo_contrastive`, which keeps PPO as the main optimizer and adds:

- a state encoder
- a goal encoder
- an auxiliary InfoNCE contrastive loss over future achieved goals

## Environment mapping

- `Reacher -> reacher`
- `Pusher Hard -> pusher_hard`
- `Humanoid -> humanoid`
- `Ant -> ant`
- `Ant U-Maze -> ant_u_maze`
- `Ant Big Maze -> ant_big_maze`
- `Ant Soccer -> ant_ball`
- `Ant Push -> ant_push`

## Local commands

Smoke examples:

```bash
python run.py ppo --env reacher --total_env_steps 1000000 --num_evals 10 --no-log-wandb
python run.py ppo_contrastive --env reacher --total_env_steps 1000000 --num_evals 10 --no-log-wandb
python3 scripts/ppo_benchmark.py --profile smoke --agent ppo --env ant --seed 0 --dry-run
```

Run one benchmark job through the helper:

```bash
python3 scripts/ppo_benchmark.py --profile pilot --agent ppo_contrastive --env ant_big_maze --seed 0
```

Exact phase commands through the helper:

```bash
python3 scripts/ppo_benchmark.py --profile smoke --agent ppo --env reacher --seed 0
python3 scripts/ppo_benchmark.py --profile pilot --agent ppo_contrastive --env ant --seed 0
python3 scripts/ppo_benchmark.py --profile final --agent ppo --env humanoid --seed 4
```

Enable wandb explicitly:

```bash
python3 scripts/ppo_benchmark.py --profile final --agent ppo --env humanoid --seed 4 --log-wandb
```

## Lighthouse commands

Create the log directory before submission:

```bash
mkdir -p /home/$USER/logs
```

Smoke run:

```bash
sbatch scripts/slurm_smoke_ppo.sh
```

Pilot benchmark:

```bash
bash scripts/submit_ppo_benchmark.sh pilot
```

Final benchmark:

```bash
bash scripts/submit_ppo_benchmark.sh final
```

Optional conda env override for all scripts:

```bash
export JAXGCRL_CONDA_ENV=my_env_name
```

The helper scripts default to local-only logging. Pass `--log-wandb` to opt in.

Benchmark helper profiles now pass `--disable-render`, so smoke, pilot, and final jobs skip HTML rollout rendering entirely. Direct `python run.py ...` invocations still use the repo default render behavior unless you pass `--disable-render`.
