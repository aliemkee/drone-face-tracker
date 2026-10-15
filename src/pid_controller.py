import time


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float = 1.0,
        integral_limit: float = 0.5,
        derivative_alpha: float = 0.15,
    ):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self._olim = abs(output_limit)
        self._ilim = abs(integral_limit)
        self._alpha = derivative_alpha
        self._integral      = 0.0
        self._prev_error    = 0.0
        self._filtered_d    = 0.0
        self._last_time     = None

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._filtered_d = 0.0
        self._last_time  = None

    def compute(self, error: float, current_time: float = None) -> float:
        now = current_time if current_time is not None else time.monotonic()
        dt  = (now - self._last_time) if self._last_time else 0.033
        dt  = max(dt, 1e-4)
        self._last_time = now

        p = self.kp * error

        self._integral = self._clamp(self._integral + error * dt, self._ilim)
        i = self.ki * self._integral

        raw_d = (error - self._prev_error) / dt
        self._filtered_d = self._alpha * self._filtered_d + (1.0 - self._alpha) * raw_d
        d = self.kd * self._filtered_d

        self._prev_error = error
        return self._clamp(p + i + d, self._olim)

    @staticmethod
    def _clamp(v, lim):
        return max(-lim, min(lim, v))
