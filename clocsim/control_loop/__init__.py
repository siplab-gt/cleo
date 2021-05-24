"""Classes and functions for constructing and configuring a :func:`~base.ControlLoop`."""

from abc import ABC, abstractmethod
from typing import Tuple, Any
from collections import deque

from .. import ControlLoop
from .delays import Delay


class LoopComponent(ABC):
    """Abstract signal processing stage or control block."""

    def __init__(self, **kwargs):
        """Construct a `LoopComponent` object.

        It's important to use `super().__init__(kwargs)` in the base class
        to use the parent-class logic here.

        Raises
        ------
        TypeError
            When `delay` is not a `Delay` object.
        """
        self.delay = kwargs.get("delay", None)
        if not isinstance(self.delay, Delay):
            raise TypeError("delay must be of the Delay class")
        self.save_history = kwargs.get("save_history", False)
        if self.save_history is True:
            self.t = []
            self.out_t = []
            self.values = []

    def process(self, input: Any, in_time_ms: float, **kwargs) -> Tuple[Any, float]:
        """Compute output and output time given input and input time.

        The user should implement :func:`~_process()`, which performs the
        computation itself without regards for the delay.

        Parameters
        ----------
        input : Any
        in_time_ms : float
        **kwargs : key-value list of arguments passed to :func:`~_process()`

        Returns
        -------
        Tuple[Any, float]
            output, out time
        """
        out = self._process(input, **kwargs)
        if self.delay is not None:
            out_time_ms = self.delay.add_delay_to_time(in_time_ms)
        else:
            out_time_ms = in_time_ms
        if self.save_history:
            self.t.append(in_time_ms)
            self.out_t.append(out_time_ms)
            self.values.append(out)
        return (out, out_time_ms)

    @abstractmethod
    def _process(self, input: Any, **kwargs) -> Any:
        """Computes output for given input.

        This is where the user will implement the desired functionality
        of the `LoopComponent` without regard for latency.

        Parameters
        ----------
        input : Any
        **kwargs : optional key-value argument pairs passed from
        :func:`~process()`. Could be used to pass in such values as
        the control loop's walltime or the measurement time for time-
        dependent functions.

        Returns
        -------
        Any
            output.
        """
        pass


class DelayControlLoop(ControlLoop):
    """
    The unit for keeping track of time in the control loop is milliseconds.
    To deal in quantities relative to seconds (e.g., defining a target firing
    rate in Hz), the component involved must make the conversion.

    For non-serial processing, the order of input/output is preserved even
    if the computed output time for a sample is sooner than that for a previous
    sample.

    Fixed sampling: on a fixed schedule no matter what
    Wait for computation sampling: Can't sample during computation. Samples ASAP
    after an over-period computation: otherwise remains on schedule.
    """

    def __init__(self, sampling_period_ms, **kwargs):
        self.out_buffer = deque([])
        self.sampling_period_ms = sampling_period_ms
        self.sampling = kwargs.get("sampling", "fixed")
        if self.sampling not in ["fixed", "wait for computation"]:
            raise ValueError("Invalid sampling scheme:", self.sampling)
        self.processing = kwargs.get("processing", "serial")
        if self.processing not in ["serial", "parallel"]:
            raise ValueError("Invalid processing scheme:", self.processing)

    def put_state(self, state_dict: dict, sample_time_ms):
        out, out_time_ms = self.compute_ctrl_signal(state_dict, sample_time_ms)
        if self.processing == "serial" and len(self.out_buffer) > 0:
            prev_out_time_ms = self.out_buffer[-1][1]
            # add delay onto the output time of the last computation
            out_time_ms = prev_out_time_ms + out_time_ms - sample_time_ms
        self.out_buffer.append((out, out_time_ms))
        self._needs_off_schedule_sample = False

    def get_ctrl_signal(self, query_time_ms):
        if len(self.out_buffer) == 0:
            return None
        next_out_signal, next_out_time_ms = self.out_buffer[0]
        if query_time_ms >= next_out_time_ms:
            self.out_buffer.popleft()
            return next_out_signal
        else:
            return None

    def _is_currently_idle(self, query_time_ms):
        return len(self.out_buffer) == 0 or self.out_buffer[0][1] <= query_time_ms

    def is_sampling_now(self, query_time_ms):
        if self.sampling == "fixed":
            if query_time_ms % self.sampling_period_ms == 0:
                return True
        elif self.sampling == "wait for computation":
            if query_time_ms % self.sampling_period_ms == 0:
                if self._is_currently_idle(query_time_ms):
                    self._needs_off_schedule_sample = False
                    return True
                else:  # if not done computing
                    self._needs_off_schedule_sample = True
                    return False
            else:
                # off-schedule, only sample if the last sampling period
                # was missed (there was an overrun)
                return self._needs_off_schedule_sample and self._is_currently_idle(
                    query_time_ms
                )
        return False

    @abstractmethod
    def compute_ctrl_signal(
        self, state_dict: dict, sample_time_ms: float
    ) -> Tuple[dict, float]:
        """Process network state to generate output to update stimulators.

        This is the function the user must implement to define the signal processing
        pipeline.

        Parameters
        ----------
        state_dict : dict
            {`recorder_name`: `state`} dictionary from :func:`~base.CLOCSimulator.get_state()`
        time_ms : float

        Returns
        -------
        Tuple[dict, float]
            {'stim_name`: `ctrl_signal`} dictionary and output time in milliseconds.
        """
        pass
