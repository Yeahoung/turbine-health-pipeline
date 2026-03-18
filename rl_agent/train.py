"""
train.py
--------
Train a PPO agent on WindTurbineEnv using stable-baselines3.

Usage:
    python train.py                   # default 200 000 steps
    python train.py --steps 500000
    python train.py --steps 100000 --seed 0
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from wind_turbine_env import WindTurbineEnv

MODEL_PATH = "models/ppo_wind_turbine"
LOG_PATH   = "logs/"


def train(total_timesteps: int = 200_000, seed: int = 42) -> PPO:
    os.makedirs("models", exist_ok=True)
    os.makedirs(LOG_PATH,  exist_ok=True)

    # Vectorised envs speed up data collection
    train_env = make_vec_env(WindTurbineEnv, n_envs=4, seed=seed)
    eval_env  = WindTurbineEnv(seed=seed + 99)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_PATH}_best",
        log_path=LOG_PATH,
        eval_freq=10_000 // 4,     # every 10 k env steps (4 envs → 2 500 calls)
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        verbose=1,
        seed=seed,
        tensorboard_log=LOG_PATH,
    )

    print(f"Training PPO for {total_timesteps:,} timesteps …")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    model.save(MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}.zip")

    train_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on WindTurbineEnv")
    parser.add_argument("--steps", type=int, default=200_000,
                        help="Total training timesteps (default: 200 000)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    train(total_timesteps=args.steps, seed=args.seed)
