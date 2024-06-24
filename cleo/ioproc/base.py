"""Basic processor and processing block definitions"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Tuple

import numpy as np
from attrs import define, field, fields
from brian2 import Quantity, ms

from cleo.base import IOProcessor
from cleo.ioproc.delays import Delay
from cleo.utilities import unit_safe_append


class ProcessingBlock(ABC):
    """Abstract signal processing stage or control block."""

    delay: Delay
    """The delay object determining compute latency for the block"""
    save_history: bool
    """Whether to record :attr:`t_in`, :attr:`t_out`, 
    and :attr:`values` with every timestep"""
    t_in: Quantity
    """The walltime the block received each input.
    Only recorded if :attr:`save_history`"""
    t_out: Quantity
    """The walltime of each of the block's outputs.
    Only recorded if :attr:`save_history`"""
    values: list[Any]
    """Each of the block's outputs.
    Only recorded if :attr:`save_history`"""

    def __init__(self, **kwargs):
        """
        It's important to use `super().__init__(**kwargs)` in the base class
        to use the parent-class logic here.

        Keyword args
        ------------
        delay : Delay
            Delay object which adds to the compute time

        Raises
        ------
        TypeError
            When `delay` is not a `Delay` object.
        """
        self.delay = kwargs.get("delay", None)
        if self.delay and not isinstance(self.delay, Delay):
            raise TypeError("delay must be of the Delay class")
        self.save_history = kwargs.get("save_history", False)
        if self.save_history is True:
            self.t_in = []
            self.t_out = []
            self.values = []

    def process(self, input: Any, t_in_ms: float, **kwargs) -> Tuple[Any, float]:
        """Computes and saves output and output time given input and input time.

        The user should implement :meth:`~compute_output()` for their child
        classes, which performs the computation itself without regards for
        timing or saving variables.

        Parameters
        ----------
        input : Any
            Data to be processed
        t_in_ms : float
            Time the block receives the input data
        **kwargs : Any
            Key-value list of arguments passed to :func:`~compute_output()`

        Returns
        -------
        Tuple[Any, float]
            output, out time in milliseconds
        """
        out = self.compute_output(input, **kwargs)
        if self.delay is not None:
            t_out_ms = t_in_ms + self.delay.compute()
        else:
            t_out_ms = t_in_ms
        if self.save_history:
            self.t_in_ms.append(t_in_ms)
            self.t_out_ms.append(t_out_ms)
            self.values.append(out)
        return (out, t_out_ms)

    @abstractmethod
    def compute_output(self, input: Any, **kwargs) -> Any:
        """Computes output for given input.

        This is where the user will implement the desired functionality
        of the `ProcessingBlock` without regard for latency.

        Parameters
        ----------
        input : Any
            Data to be processed. Passed in from :meth:`process`.
        **kwargs : Any
            optional key-value argument pairs passed from
            :meth:`process`. Could be used to pass in such values as
            the IO processor's walltime or the measurement time for time-
            dependent functions.

        Returns
        -------
        Any
            output
        """
        pass


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
