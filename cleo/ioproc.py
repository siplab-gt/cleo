"""Basic processor definitions and control/estimation functions"""
from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import Tuple

import numpy as np
from attrs import define, field, fields
from brian2 import Quantity, ms
from jaxtyping import UInt

from cleo.base import IOProcessor
from cleo.utilities import unit_safe_append


@define
class LatencyIOProcessor(IOProcessor):
    """IOProcessor capable of delivering stimulation some time after measurement.

    Note
    ----
    It doesn't make much sense to combine parallel computation
    with "when idle" sampling, because "when idle" sampling only produces
    one sample at a time to process.
    """

    t_samp: Quantity = field(factory=lambda: [] * ms, init=False, repr=False)
    """Record of sampling times---each time :meth:`~put_state` is called."""

    sampling: str = field(default="fixed")
    """Sampling scheme: "fixed" or "when idle".
    
    "fixed" sampling means samples are taken on a fixed schedule,
    with no exceptions.
    
        

    "when idle" sampling means no samples are taken before the previous
    sample's output has been delivered. A sample is taken ASAP
    after an over-period computation: otherwise remains on schedule.
    """

    @sampling.validator
    def _validate_sampling(self, attribute, value):
        if value not in ["fixed", "when idle"]:
            raise ValueError("Invalid sampling scheme:", value)

    processing: str = field(default="parallel")
    """Processing scheme: "serial" or "parallel".

    "parallel" computes the output time by adding the delay for a sample
    onto the sample time, so if the delay is 2 ms, for example, while the
    sample period is only 1 ms, some of the processing is happening in
    parallel. Output order matches input order even if the computed
    output time for a sample is sooner than that for a previous
    sample.

    "serial" computes the output time by adding the delay for a sample
    onto the output time of the previous sample, rather than the sampling
    time. Note this may be of limited
    utility because it essentially means the *entire* round trip
    cannot be in parallel at all. More realistic is that simply
    each block or phase of computation must be serial. If anyone
    cares enough about this, it will have to be implemented in the
    future.
    """

    @processing.validator
    def _validate_processing(self, attribute, value):
        if value not in ["serial", "parallel"]:
            raise ValueError("Invalid processing scheme:", value)

    out_buffer: deque[Tuple[dict, float]] = field(factory=deque, init=False, repr=False)
    """
        "serial" computes the output time by adding the delay for a sample
        onto the output time of the previous sample, rather than the sampling
        time. Note this may be of limited
        utility because it essentially means the *entire* round trip
        cannot be in parallel at all. More realistic is that simply
        each block or phase of computation must be serial. If anyone
        cares enough about this, it will have to be implemented in the
        future.

        Note
        ----
        It doesn't make much sense to combine parallel computation
        with "when idle" sampling, because "when idle" sampling only produces
        one sample at a time to process.

        Raises
        ------
        ValueError
            For invalid `sampling` or `processing` kwargs 
        """

    def put_state(self, state_dict: dict, t_samp: Quantity):
        self.t_samp = unit_safe_append(self.t_samp, t_samp)
        out, t_out = self.process(state_dict, t_samp)
        if self.processing == "serial" and len(self.out_buffer) > 0:
            prev_t_out = self.out_buffer[-1][1]
            # add delay onto the output time of the last computation
            t_out = prev_t_out + t_out - t_samp
        self.out_buffer.append((out, t_out))
        self._needs_off_schedule_sample = False

    def get_ctrl_signals(self, t_query):
        if len(self.out_buffer) == 0:
            return {}
        next_out_signal, next_t_out = self.out_buffer[0]
        if t_query >= next_t_out:
            self.out_buffer.popleft()
            return next_out_signal
        else:
            return {}

    def _is_currently_idle(self, t_query):
        return len(self.out_buffer) == 0 or self.out_buffer[0][1] <= t_query

    def is_sampling_now(self, t_query):
        resid_ms = np.round((t_query % self.sample_period) / ms, 6)
        if self.sampling == "fixed":
            if np.isclose(resid_ms, 0) or np.isclose(
                resid_ms, np.round(self.sample_period / ms, 6)
            ):
                return True
        elif self.sampling == "when idle":
            if np.isclose(resid_ms, 0):
                if self._is_currently_idle(t_query):
                    self._needs_off_schedule_sample = False
                    return True
                else:  # if not done computing
                    self._needs_off_schedule_sample = True
                    return False
            else:
                # off-schedule, only sample if the last sampling period
                # was missed (there was an overrun)
                return self._needs_off_schedule_sample and self._is_currently_idle(
                    t_query
                )
        return False

    @abstractmethod
    def process(self, state_dict: dict, t_samp: Quantity) -> Tuple[dict, Quantity]:
        """Process network state to generate output to update stimulators.

        This is the function the user must implement to define the signal processing
        pipeline.

        Parameters
        ----------
        state_dict : dict
            {`recorder_name`: `state`} dictionary from :func:`~cleo.CLSimulator.get_state()`
        t_samp : Quantity
            The time at which the sample was taken.

        Returns
        -------
        Tuple[dict, Quantity]
            {'stim_name': `ctrl_signal`} dictionary and output time (including unit).
        """
        pass

    def _base_reset(self):
        self.t_samp = fields(type(self)).t_samp.default.factory()
        self.out_buffer = fields(type(self)).out_buffer.default.factory()
        self._needs_off_schedule_sample = False


class RecordOnlyProcessor(LatencyIOProcessor):
    """Take samples without performing any control.

    Use this if all you are doing is recording."""

    def __init__(self, sample_period, **kwargs):
        super().__init__(sample_period, **kwargs)

    def process(self, state_dict: dict, sample_time: float) -> Tuple[dict, float]:
        return ({}, sample_time)


def exp_firing_rate_estimate(
    spike_counts: UInt[np.ndarray, "num_spike_sources"],
    dt: Quantity,
    prev_rate: Quantity,
    tau: Quantity,
) -> Quantity:
    """Estimate firing rate with a recursive exponential filter.

    Parameters
    ----------
    spike_counts: np.ndarray
        n-length vector of spike counts
    dt: Quantity
        Time since last measurement (with Brian temporal unit)
    prev_rate: Quantity
        n-length vector of previously estimated firing rates
    tau: Quantity
        Time constant of exponential filter (with Brian temporal unit)

    Returns
    -------
    Quantity
        n-length vector of estimated firing rates (with Brian units)
    """
    alpha = np.exp(-dt / tau)
    return prev_rate * alpha + (1 - alpha) * spike_counts / dt


def pi_ctrl(
    measurement: float,
    reference: float,
    integ_error: float,
    dt: Quantity,
    Kp: float,
    Ki: Quantity = 0 / ms,
):
    error = reference - measurement
    integ_error += error * dt
    return Kp * error + Ki * integ_error, integ_error
