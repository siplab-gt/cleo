from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from attrs import define, field
from nptyping import NDArray

from brian2 import np


class ImagingNoise(ABC):
    @abstractmethod
    def init_for_ng(
        self,
        ng,
        i_targets,
        scope_location,
        scope_direction,
        scope_focus_depth,
        **injct_kwargs,
    ):
        """Could be useful to perform spatially dependent calculations
        just once at the beginning"""
        pass

    @abstractmethod
    def compute(self, t_ms) -> NDArray[(Any,), float]:
        """Returns a noise array in the order neuron groups/targets were received"""
        pass


@define(eq=False)
class UniformGaussianNoise(ImagingNoise):
    sigma: float = None
    _sigma_array: NDArray[(Any,), float] = field(
        init=False, factory=lambda: np.array([])
    )

    def init_for_ng(
        self,
        ng,
        i_targets,
        scope_location,
        scope_direction,
        scope_focus_depth,
        **injct_kwargs,
    ):
        # keyword args: sigma_ugn
        sigma = injct_kwargs.get("sigma_ugn", self.sigma)
        if sigma is None:
            raise ValueError("sigma must be specified in either init or inject")
        self._sigma_array = np.concatenate(
            [self._sigma_array, np.full(len(i_targets), sigma)]
        )

    def compute(self, t_ms) -> list[np.ndarray]:
        return [np.random.normal(0, self.sigma, len(t_ms))]
