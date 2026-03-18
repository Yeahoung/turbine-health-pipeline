"""
rule_based_agent.py
-------------------
Simple threshold-based pitch controller used as the heuristic baseline.

Logic:
  - Below rated wind speed: hold fine pitch (FINE_PITCH) to maximise capture.
  - Above rated wind speed: linearly feather the blade to limit power and load.
  - Deadband prevents hunting (constant small oscillations).

The predict() signature mirrors the stable-baselines3 policy interface so
the same evaluation loop works for both agents.
"""

import numpy as np


class RuleBasedAgent:
    """Deterministic rule-based pitch controller."""

    RATED_WIND  = 12.0   # m/s – must match WindTurbineEnv
    FINE_PITCH  = 5.0    # degrees – fine-pitch target below rated
    DEADBAND    = 1.5    # degrees – no-action zone around target

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        Parameters
        ----------
        obs : np.ndarray  shape (5,) or (N, 5)
            [wind_speed, wind_trend, pitch_angle, power, load]
        deterministic : bool  (ignored; always deterministic)

        Returns
        -------
        action : np.ndarray  shape (1,)
            0 = pitch down, 1 = hold, 2 = pitch up
        state : None
        """
        # Support both single obs (1-D) and batched obs (2-D)
        obs = np.atleast_2d(obs)
        actions = np.array([self._select_action(o) for o in obs])
        return actions, None

    # ------------------------------------------------------------------

    def _select_action(self, obs: np.ndarray) -> int:
        wind_speed, _wind_trend, pitch_angle, _power, _load = obs

        target_pitch = self._target_pitch(wind_speed)
        error = target_pitch - pitch_angle

        if error < -self.DEADBAND:
            return 0   # pitch down
        if error > self.DEADBAND:
            return 2   # pitch up
        return 1       # hold

    def _target_pitch(self, wind_speed: float) -> float:
        if wind_speed <= self.RATED_WIND:
            return self.FINE_PITCH
        # Linear feathering above rated; caps at 45 °
        excess = wind_speed - self.RATED_WIND
        return min(self.FINE_PITCH + excess * 4.0, 45.0)
