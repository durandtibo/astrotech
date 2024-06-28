r"""Contain a loss function that returns an empty dictionary."""

from __future__ import annotations

__all__ = ["NoLoss"]

from typing import TYPE_CHECKING

from torch.nn import Module

if TYPE_CHECKING:
    from torch import Tensor


class NoLoss(Module):
    r"""Implement a loss function that returns an empty dictionary.

    THIS LOSS CANNOT BE USED TO TRAIN A MODEL.

    This loss is designed to be used when it is not possible to compute
    a loss i.e. at inference when the target is not available.

    Example usage:

    ```pycon

    >>> import torch
    >>> from astrotech.model.criteria import NoLoss
    >>> criterion = NoLoss()
    >>> criterion
    NoLoss()
    >>> out = criterion({}, {})
    >>> out
    {}

    ```
    """

    def forward(self, net_out: dict, batch: dict) -> dict[str, Tensor]:  # noqa: ARG002
        return {}
