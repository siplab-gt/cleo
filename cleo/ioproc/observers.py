from typing import Any
import numpy as np
from nptyping import NDArray
from cleo.ioproc.base import ProcessingBlock


class FiringRateEstimator(ProcessingBlock):
    """Exponential filter to estimate firing rate.

    Requires `sample_time_ms` kwarg when process is called.
    """

    def __init__(self, tau_ms: float, sample_period_ms: float, **kwargs):
        """
        Parameters
        ----------
        tau_ms : float
            Time constant of filter
        sample_period_ms : float
            Sampling period in milliseconds
        """
        super().__init__(**kwargs)
        self.tau_s = tau_ms / 1000
        self.T_s = sample_period_ms / 1000
        self.alpha = np.exp(-sample_period_ms / tau_ms)
        self.prev_rate = None
        self.prev_time_ms = None

    def compute_output(
        self, input: NDArray[(Any,), np.uint], **kwargs
    ) -> NDArray[(Any,), float]:
        """Estimate firing rate given past and current spikes.

        Parameters
        ----------
        input: NDArray[(n,), np.uint]
            n-length vector of spike counts

        Keyword args
        ------------
        sample_time_ms: float
            Time measurement was taken in milliseconds

        Returns
        -------
        NDArray[(n,), float]
            n-length vector of firing rates
        """
        time_ms = kwargs["sample_time_ms"]
        if self.prev_rate is None:
            self.prev_rate = np.zeros(input.shape)
        if self.prev_time_ms is None:
            self.prev_time_ms = time_ms - self.T_s * 1000

        intersample_period_s = (time_ms - self.prev_time_ms) / 1000
        alpha = np.exp(-intersample_period_s / self.tau_s)
        curr_rate = self.prev_rate * alpha + (1 - alpha) * input / intersample_period_s
        self.prev_rate = curr_rate
        self.prev_time_ms = time_ms
        return curr_rate
