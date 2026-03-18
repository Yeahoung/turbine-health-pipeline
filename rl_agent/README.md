# RL Wind Turbine Pitch Control — MVP Demo

A minimal reinforcement learning demo showing that a PPO agent can outperform
a simple rule-based heuristic when controlling the blade pitch angle of a
simulated wind turbine.

> **Note:** This is a conceptual competition project. The simulation is
> intentionally simplified — it is not a real turbine controller.

---

## Project structure

```
Scottish_Power_rl/
├── wind_turbine_env.py   # Gymnasium environment
├── rule_based_agent.py   # Heuristic baseline controller
├── train.py              # PPO training script
├── evaluate.py           # Evaluation & comparison plots
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies (virtual environment recommended)
pip install -r requirements.txt

# 2. Train the PPO agent  (~2–5 min on CPU, 200 k steps)
python train.py

# 3. Evaluate and compare against the rule-based baseline
python evaluate.py
```

Optional flags:

```bash
python train.py    --steps 500000 --seed 0
python evaluate.py --episodes 30
python evaluate.py --no-plots          # print metrics only
```

---

## Environment — `WindTurbineEnv`

| Property | Detail |
|---|---|
| **State** (5 floats) | `[wind_speed, wind_trend, pitch_angle, power, load]` |
| **Actions** (Discrete 3) | `0` pitch down · `1` hold · `2` pitch up (±3 ° per step) |
| **Episode length** | 500 steps |
| **Wind model** | random walk + slow trend drift + occasional gusts (3 %) |

### Operating regions

| Region | Wind | Objective |
|---|---|---|
| Below cut-in / above cut-out | < 3 m/s or > 25 m/s | No generation |
| **Below rated** | 3 – 12 m/s | Fine pitch → maximise capture |
| **Above rated** | 12 – 25 m/s | Feather blade → limit power & load |

### Reward function

```
R = 2.0 × power  −  1.5 × load  −  0.1 × |Δpitch| / pitch_step
```

The agent is rewarded for generating power, penalised for structural load,
and slightly penalised for unnecessary pitch activity.

---

## Rule-based baseline — `RuleBasedAgent`

A threshold controller that follows a fixed schedule:

- **Below rated:** target pitch = 5 °  (fine pitch, full capture)
- **Above rated:** target pitch increases linearly with wind speed, capped at 45 °
- 1.5 ° deadband prevents unnecessary hunting

---

## Training — `train.py`

Uses `stable-baselines3` PPO with:

| Hyperparameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁴ |
| Steps per rollout | 1 024 per env |
| Batch size | 64 |
| Epochs per update | 10 |
| Discount γ | 0.99 |
| Parallel envs | 4 |

A periodic `EvalCallback` saves the best checkpoint to `models/ppo_wind_turbine_best/`.

---

## Evaluation — `evaluate.py`

Runs both agents over `--episodes` episodes (same seeds for a fair comparison)
and prints a metric summary, then produces two plots saved to `results/`:

| Plot | Description |
|---|---|
| `episode_comparison.png` | Time-series of wind, pitch, power, load for one shared episode |
| `reward_distribution.png` | Histogram of total episode rewards for both agents |

---

## Expected results

After 200 k training steps the PPO agent typically achieves a **10–25 % higher
mean episode reward** than the rule-based baseline, primarily by learning smoother
pitch transitions and more precise feathering above rated wind speed.

Individual-run results will vary with random seeds; re-run `train.py` with a
different `--seed` if the agent has not converged.

---

## File-level design notes

- **`wind_turbine_env.py`** — self-contained; no external data files.
  Equations are deliberately simplified (no full blade-element momentum model).
- **`rule_based_agent.py`** — `predict()` mirrors the SB3 policy interface so
  the same `run_episode()` loop works for both agents.
- **`train.py`** — vectorised training via `make_vec_env`; callbacks handle
  best-model checkpointing.
- **`evaluate.py`** — deterministic rollouts with fixed seeds ensure
  reproducible comparisons.
