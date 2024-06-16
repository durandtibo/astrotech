r"""Contain the metric base class."""

from __future__ import annotations

__all__ = ["BaseMetric", "EmptyMetricError", "setup_metric"]

import logging
from abc import abstractmethod
from typing import Any
from unittest.mock import Mock

from objectory import AbstractFactory
from torch.nn import Module

from astrotech.utils.factory import str_target_object

logger = logging.getLogger(__name__)

BaseEngine = Mock()  # TODO (tibo): remove after engine is implemented  # noqa: TD003


# TODO (tibo): add example to the docstrings  # noqa: TD003


class BaseMetric(Module, metaclass=AbstractFactory):
    r"""Defines the base class for the metric.

    This class is used to register the metric using the metaclass
    factory. Child classes must implement the following methods:
        - ``attach``
        - ``forward``
        - ``reset``
        - ``value``
    """

    @abstractmethod
    def attach(self, engine: BaseEngine) -> None:
        r"""Attach the current metric to an engine.

        This method can be used to:

            - add event handler to the engine
            - set up history trackers

        Args:
            engine: The engine.
        """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> dict | None:
        r"""Update the metric given a mini-batch of examples.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    @abstractmethod
    def reset(self) -> None:
        r"""Reset the internal state of the metric."""

    @abstractmethod
    def value(self, engine: BaseEngine | None = None) -> dict:
        r"""Evaluate the metric and log the results given the current
        metric state.

        Args:
            engine: The engine.

        Returns:
             The results of the metric.
        """


class EmptyMetricError(Exception):
    r"""Raised when you try to evaluate an empty metric."""


def setup_metric(metric: BaseMetric | dict) -> BaseMetric:
    r"""Set up the metric.

    Args:
        metric: The metric or its configuration.

    Returns:
        The instantiated metric.
    """
    if isinstance(metric, dict):
        logger.info(f"Initializing a metric from its configuration... {str_target_object(metric)}")
        metric = BaseMetric.factory(**metric)
    if not isinstance(metric, Module):
        logger.warning(f"metric is not a `torch.nn.Module` (received: {type(metric)})")
    return metric
