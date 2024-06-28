r"""Contain a criterion that has a different behavior based on the mode
(training or evaluation)."""

from __future__ import annotations

__all__ = ["ModeLoss"]

from typing import TYPE_CHECKING

from karbonn.utils import setup_module
from torch.nn import Module

if TYPE_CHECKING:
    from torch import Tensor


class ModeLoss(Module):
    r"""Implement a criterion that has a different behavior based on the
    mode (training or evaluation).

    Args:
        train: The criterion (or its configuration) to use during
            training.
        eval: The criterion (or its configuration) to use during
            evaluation.

    Example usage:

    ```pycon

    >>> import torch
    >>> from astrotech.model.criteria import ModeLoss, Loss
    >>> criterion = ModeLoss(train=Loss(torch.nn.MSELoss()), eval=Loss(torch.nn.L1Loss()))
    >>> criterion
    ModeLoss(
      (train_criterion): Loss(
        (prediction): prediction
        (target): target
        (weight): 1.0
        (criterion): MSELoss()
      )
      (eval_criterion): Loss(
        (prediction): prediction
        (target): target
        (weight): 1.0
        (criterion): L1Loss()
      )
    )
    >>> # training mode
    >>> criterion.train()
    >>> loss = criterion(
    ...     net_out={"prediction": torch.zeros(2, 4)}, batch={"target": 2 * torch.ones(2, 4)}
    ... )
    >>> loss
    {'loss': tensor(4.)}
    >>> # evaluation mode
    >>> criterion.eval()
    >>> loss = criterion(
    ...     net_out={"prediction": torch.zeros(2, 4)}, batch={"target": 2 * torch.ones(2, 4)}
    ... )
    >>> loss
    {'loss': tensor(2.)}

    ```
    """

    def __init__(self, train: Module | dict, eval: Module | dict) -> None:  # noqa: A002
        super().__init__()
        # It is not possible to use 'train' or 'eval' because these
        # names are already used in ``torch.nn.Module``
        self.train_criterion = setup_module(train)
        self.eval_criterion = setup_module(eval)

    def forward(self, net_out: dict, batch: dict) -> dict[str, Tensor]:
        r"""Return the loss value given the network output and the batch.

        Args:
            net_out: The network output which contains the prediction.
            batch: The batch which contains the target.

        Returns:
            A dict with the loss value.
        """
        if self.training:
            return self.train_criterion(net_out, batch)
        return self.eval_criterion(net_out, batch)
