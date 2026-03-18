"""
wind_turbine_env.py
-------------------
Simplified wind turbine pitch control environment built with Gymnasium.

Observation (5 floats):
    [wind_speed, wind_trend, pitch_angle, power_output, structural_load]

Actions (Discrete 3):
    0 = decrease pitch by PITCH_STEP degrees
    1 = hold
    2 = increase pitch by PITCH_STEP degrees

Reward:
    power_reward  – encourage high normalised power output
    load_penalty  – discourage high structural load
    pitch_penalty – discourage unnecessary pitch activity
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WindTurbineEnv(gym.Env):
    metadata = {"render_modes": []}

    # --- Turbine parameters ---
    RATED_WIND       = 12.0   # m/s  rated wind speed
    CUT_IN_WIND      = 3.0    # m/s  cut-in speed
    CUT_OUT_WIND     = 25.0   # m/s  cut-out speed
    RATED_POWER      = 1.0    # normalised rated power

    PITCH_MIN        = 0.0    # degrees
    PITCH_MAX        = 90.0   # degrees
    PITCH_STEP       = 3.0    # degrees per action

    FINE_PITCH       = 5.0    # optimal pitch below rated (degrees)

    # --- Reward weights ---
    W_POWER  = 2.0
    W_LOAD   = 1.5
    W_PITCH  = 0.1

    MAX_EPISODE_STEPS = 500

    def __init__(self, seed: int | None = None):
        super().__init__()

        self.action_space = spaces.Discrete(3)

        low  = np.array([0.0, -1.0,  0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([30.0, 1.0, 90.0, 1.5, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._step = 0
        self.wind_speed  = 0.0
        self.wind_trend  = 0.0
        self.pitch_angle = 0.0
        self.power       = 0.0
        self.load        = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step       = 0
        self.wind_speed  = float(self._rng.uniform(5.0, 15.0))
        self.wind_trend  = float(self._rng.uniform(-0.2, 0.2))
        self.pitch_angle = self.FINE_PITCH
        self.power, self.load = self._compute_power_load()

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Apply pitch adjustment
        delta = (int(action) - 1) * self.PITCH_STEP          # -3, 0, or +3
        self.pitch_angle = float(
            np.clip(self.pitch_angle + delta, self.PITCH_MIN, self.PITCH_MAX)
        )

        # 2. Advance wind simulation
        self._evolve_wind()

        # 3. Physics
        self.power, self.load = self._compute_power_load()

        # 4. Scalar reward
        reward = self._compute_reward(abs(delta))

        self._step += 1
        truncated  = self._step >= self.MAX_EPISODE_STEPS
        terminated = False

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Wind evolution
    # ------------------------------------------------------------------

    def _evolve_wind(self):
        # Slowly drifting wind trend
        self.wind_trend += float(self._rng.normal(0.0, 0.05))
        self.wind_trend  = float(np.clip(self.wind_trend, -0.5, 0.5))

        # Random-walk wind speed
        noise = float(self._rng.normal(0.0, 0.3))
        self.wind_speed += self.wind_trend + noise

        # Occasional gust (3 % of steps)
        if self._rng.random() < 0.03:
            sign = float(self._rng.choice([-1.0, 1.0]))
            gust = sign * float(self._rng.uniform(2.0, 5.0))
            self.wind_speed += gust

        self.wind_speed = float(np.clip(self.wind_speed, 1.0, 28.0))

    # ------------------------------------------------------------------
    # Simplified turbine physics
    # ------------------------------------------------------------------

    def _compute_power_load(self) -> tuple[float, float]:
        w = self.wind_speed
        p = self.pitch_angle

        # --- Outside operating envelope ---
        if w < self.CUT_IN_WIND or w > self.CUT_OUT_WIND:
            return 0.0, 0.1

        # --- Below rated: maximise capture ---
        if w <= self.RATED_WIND:
            # Gaussian Cp curve centred on fine pitch
            cp_factor = np.exp(-0.02 * (p - self.FINE_PITCH) ** 2)
            power = float(np.clip((w / self.RATED_WIND) ** 3 * cp_factor, 0.0, 1.5))
            load  = 0.3 * (w / self.RATED_WIND) + 0.05 * abs(p - self.FINE_PITCH) / 45.0

        # --- Above rated: feather to limit power and load ---
        else:
            # Ideal pitch rises linearly with excess wind speed
            ideal_pitch = self.FINE_PITCH + (w - self.RATED_WIND) * 4.0
            ideal_pitch = min(ideal_pitch, 45.0)

            under_pitch_error = max(0.0, ideal_pitch - p)   # catching too much wind
            over_pitch_error  = max(0.0, p - ideal_pitch)   # spilling too much power

            power = float(np.clip(1.0 - 0.04 * over_pitch_error, 0.0, 1.5))
            load  = (
                0.3
                + 0.5 * (w - self.RATED_WIND) / (self.CUT_OUT_WIND - self.RATED_WIND)
                + 0.5 * under_pitch_error / 45.0
            )

        return float(np.clip(power, 0.0, 1.5)), float(np.clip(load, 0.0, 2.0))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, pitch_change_magnitude: float) -> float:
        power_reward  = self.W_POWER * self.power
        load_penalty  = self.W_LOAD  * self.load
        # Normalise pitch activity to [0, 1]
        pitch_penalty = self.W_PITCH * (pitch_change_magnitude / self.PITCH_STEP)
        return power_reward - load_penalty - pitch_penalty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self.wind_speed, self.wind_trend, self.pitch_angle, self.power, self.load],
            dtype=np.float32,
        )
