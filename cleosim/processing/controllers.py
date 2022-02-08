"""Contains a basic PI controller"""
from __future__ import annotations
from cleosim.processing import ProcessingBlock
from typing import Any


class Controller(ProcessingBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PIController(Controller):
    """Simple PI controller.

    :meth:`_process` requires a ``sample_time_ms`` keyword argument."""

    ref_signal: callable[[float], Any]
    """"Callable returning the target as a function of time in ms"""

    def __init__(
        self,
        ref_signal: callable,
        Kp: float,
        Ki: float = 0,
        sample_period_ms: float = 0,
        **kwargs: Any
    ):
        """
        Parameters
        ----------
        ref_signal : callable
            Must return the target as a function of time in ms
        Kp : float
            Gain on the proportional error
        Ki : float, optional
            Gain on the integral error, by default 0
        sample_period_ms : float, optional
            Rate at which processor takes samples, by default 0.
            Only used to compute integrated error on first sample
        """
        super().__init__(**kwargs)
        self.ref_signal = ref_signal
        self.Kp = Kp
        self.Ki = Ki
        self.sample_period_ms = sample_period_ms
        self.integrated_error = 0
        self.prev_time_ms = None

    def _process(self, input, **kwargs):
        time_ms = kwargs["sample_time_ms"]
        if self.prev_time_ms is None:
            self.prev_time_ms = time_ms - self.sample_period_ms
        intersample_period_s = (time_ms - self.prev_time_ms) / 1000
        error = self.ref_signal(time_ms) - input
        self.integrated_error += error * intersample_period_s
        self.prev_time_ms = time_ms
        return self.Kp * error + self.Ki * self.integrated_error
