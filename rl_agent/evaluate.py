"""
evaluate.py
-----------
Compare the trained PPO agent against the rule-based baseline.

Usage:
    python evaluate.py                        # uses saved model
    python evaluate.py --episodes 30
    python evaluate.py --no-plots             # metrics only

Outputs:
    results/episode_comparison.png    – time-series for one episode
    results/reward_distribution.png   – histogram over many episodes
    Printed summary table
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from wind_turbine_env import WindTurbineEnv
from rule_based_agent import RuleBasedAgent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "models", "ppo_wind_turbine")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# -----------------------------------------------------------------------
# Episode runner
# -----------------------------------------------------------------------

def run_episode(env: WindTurbineEnv, agent, seed: int | None = None) -> tuple[float, dict]:
    """Run a single episode; return (total_reward, trajectory_dict)."""
    obs, _ = env.reset(seed=seed)
    done   = False
    total_reward = 0.0
    traj = {"wind": [], "pitch": [], "power": [], "load": [], "reward": []}

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        total_reward += reward

        traj["wind"].append(float(obs[0]))
        traj["pitch"].append(float(obs[2]))
        traj["power"].append(float(obs[3]))
        traj["load"].append(float(obs[4]))
        traj["reward"].append(float(reward))

    return total_reward, traj


# -----------------------------------------------------------------------
# Multi-episode evaluation
# -----------------------------------------------------------------------

def evaluate_agent(env: WindTurbineEnv, agent, n_episodes: int, label: str) -> np.ndarray:
    rewards = []
    for ep in range(n_episodes):
        total_r, _ = run_episode(env, agent, seed=ep * 13)
        rewards.append(total_r)

    rewards = np.array(rewards)
    mean_power = []
    mean_load  = []
    for ep in range(n_episodes):
        _, traj = run_episode(env, agent, seed=ep * 13)
        mean_power.append(np.mean(traj["power"]))
        mean_load.append(np.mean(traj["load"]))

    print(
        f"  [{label:>12s}]  "
        f"reward {rewards.mean():7.2f} ± {rewards.std():.2f}  |  "
        f"power {np.mean(mean_power):.3f}  |  "
        f"load {np.mean(mean_load):.3f}"
    )
    return rewards


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------

def plot_episode(traj_rl: dict, traj_rule: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = f"{RESULTS_DIR}/episode_comparison.png"

    steps   = range(len(traj_rl["wind"]))
    metrics = [
        ("wind",   "Wind Speed (m/s)"),
        ("pitch",  "Pitch Angle (°)"),
        ("power",  "Power (norm.)"),
        ("load",   "Structural Load (norm.)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        "Wind Turbine Pitch Control — PPO vs Rule-Based (single episode)",
        fontsize=13, y=1.01,
    )

    for (key, ylabel), ax in zip(metrics, axes):
        ax.plot(steps, traj_rl[key],   label="PPO",        color="steelblue", lw=1.5)
        ax.plot(steps, traj_rule[key], label="Rule-Based", color="coral",     lw=1.5, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    return fig


def plot_reward_distribution(rewards_rl: np.ndarray, rewards_rule: np.ndarray) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = f"{RESULTS_DIR}/reward_distribution.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 15

    ax.hist(rewards_rule, bins=bins, alpha=0.55, label="Rule-Based",
            color="coral", edgecolor="white")
    ax.hist(rewards_rl,   bins=bins, alpha=0.55, label="PPO",
            color="steelblue", edgecolor="white")

    ax.axvline(rewards_rule.mean(), color="coral",     linestyle="--", lw=2,
               label=f"Rule mean: {rewards_rule.mean():.1f}")
    ax.axvline(rewards_rl.mean(),   color="steelblue", linestyle="--", lw=2,
               label=f"PPO  mean: {rewards_rl.mean():.1f}")

    ax.set_title("Episode Reward Distribution — PPO vs Rule-Based", fontsize=12)
    ax.set_xlabel("Total Episode Reward")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    return fig


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(n_episodes: int = 20, show_plots: bool = True) -> None:
    env = WindTurbineEnv()

    # Load trained model (fall back to untrained if not found)
    if os.path.exists(f"{MODEL_PATH}.zip"):
        ppo_agent = PPO.load(MODEL_PATH)
        print(f"Loaded PPO model from {MODEL_PATH}.zip\n")
    else:
        print(
            f"[WARNING] No trained model found at {MODEL_PATH}.zip.\n"
            "Run `python train.py` first for meaningful results.\n"
            "Using an untrained PPO for demonstration only.\n"
        )
        ppo_agent = PPO("MlpPolicy", env, verbose=0)

    rule_agent = RuleBasedAgent()

    # --- Quantitative comparison ---
    print("=" * 65)
    print(f"Evaluation over {n_episodes} episodes")
    print("=" * 65)
    rewards_rl   = evaluate_agent(env, ppo_agent,  n_episodes, "PPO")
    rewards_rule = evaluate_agent(env, rule_agent, n_episodes, "Rule-Based")

    delta = rewards_rl.mean() - rewards_rule.mean()
    pct   = delta / abs(rewards_rule.mean()) * 100
    print("-" * 65)
    print(f"  PPO improvement over rule-based: {delta:+.2f}  ({pct:+.1f} %)")
    print("=" * 65)

    # --- Qualitative episode comparison ---
    if show_plots:
        print("\nGenerating plots …")
        _, traj_rl   = run_episode(env, ppo_agent,  seed=42)
        _, traj_rule = run_episode(env, rule_agent, seed=42)
        plot_episode(traj_rl, traj_rule)
        plot_reward_distribution(rewards_rl, rewards_rule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO vs rule-based baseline")
    parser.add_argument("--episodes",  type=int,  default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--no-plots",  dest="plots", action="store_false",
                        help="Skip generating plots")
    args = parser.parse_args()

    main(n_episodes=args.episodes, show_plots=args.plots)
