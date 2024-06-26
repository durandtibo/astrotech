r"""Contain a criterion wrapper to make PyTorch criteria compatible with
``astrotech.model.Model`` and padded sequences."""

from __future__ import annotations

__all__ = ["PaddedSequenceLoss"]


import torch
from coola.utils import str_mapping
from karbonn.utils import setup_module
from torch import Tensor
from torch.nn import Module

from astrotech import constants as ct


class PaddedSequenceLoss(Module):
    r"""Implement a wrapper to adapt PyTorch loss function (a.k.a.
    criterion) to deal with padded sequential inputs.

    This class works for most of the PyTorch criterion that has two
    inputs: prediction and target. It assumes that the valid time
    steps of the sequence are indicated with a mask. This loss
    function should have at least two inputs:

        - the prediction which is a ``torch.Tensor`` of shape
            ``(sequence_length, batch_size, *)`` or
            ``(batch_size, sequence_length, *)`` where ``*`` means
            any number of additional dimensions. This tensor is
            converted to a tensor of shape
            ``(sequence_length * batch_size, *)`` and then feeds to
            the PyTorch loss function.
        - the target which is a ``torch.Tensor`` of shape
            ``(sequence_length, batch_size, *)`` or
            ``(batch_size, sequence_length, *)`` where ``*`` means
            any number of additional dimensions.
            This tensor is converted to a tensor of shape
            ``(sequence_length * batch_size, *)`` and then feeds to
            the PyTorch loss function.

    The input mask is optional. If no mask is provided, all the steps
    are considered as valid. The mask is a ``torch.Tensor`` of shape
    ``(sequence_length, batch_size)`` or
    ``(batch_size, sequence_length)``. The type of the tensor can be
    ``torch.int`` or``torch.long`` or``torch.float`` or ``torch.bool``
    with the following values:

        - valid value: ``True`` or ``1`` if ``valid_value=True``,
            otherwise ``False`` or ``0``.
        - invalid value: ``False`` or ``0`` if ``valid_value=True``,
            otherwise ``True`` or ``1``.

    Note that this class may not be compatible which any PyTorch
    criterion. However, you should be able to adapt this
    implementation for your use-case.

    Args:
        criterion: The loss function or its configuration.
        prediction_key: The prediction key.
        target_key: The target key.
        mask_key: The mask key.
        valid_value: Indicates the valid values in the mask.
            If ``True``, the valid values are indicated by a ``True``
            in the mask. If ``False``, the valid values are indicated
            by a ``False`` in the mask.
        mask_in_batch: Indicates if the mask is in ``batch`` or
            ``net_out``. If ``True``, the mask is taken from the input
            ``batch``, otherwise it is taken from the input
            ``net_out``.

    Example usage:

    ```pycon

    >>> from torch import nn
    >>> from astrotech.model.criteria import PaddedSequenceLoss
    >>> # Init with a nn.Module
    >>> criterion = PaddedSequenceLoss(criterion=nn.MSELoss())
    >>> criterion
    PaddedSequenceLoss(
      (prediction): prediction
      (target): target
      (mask): mask
      (weight): 1.0
      (criterion): MSELoss()
    )
    >>> # Init with a config
    >>> criterion = PaddedSequenceLoss(criterion={"_target_": "torch.nn.MSELoss"})
    >>> criterion
    PaddedSequenceLoss(
      (prediction): prediction
      (target): target
      (mask): mask
      (weight): 1.0
      (criterion): MSELoss()
    )
    >>> # Customize keys.
    >>> criterion = PaddedSequenceLoss(
    ...     criterion=nn.MSELoss(),
    ...     prediction_key="my_prediction",
    ...     target_key="my_target",
    ...     mask_key="my_mask",
    ... )
    >>> criterion
    PaddedSequenceLoss(
      (prediction): my_prediction
      (target): my_target
      (mask): my_mask
      (weight): 1.0
      (criterion): MSELoss()
    )
    >>> net_out = {"my_prediction": torch.randn(2, 4)}
    >>> batch = {"my_target": torch.randn(2, 4), "my_mask": torch.ones(2, 4)}
    >>> loss = criterion(net_out, batch)
    >>> loss
    {'loss': tensor(...)}

    ```
    """

    def __init__(
        self,
        criterion: Module | dict,
        prediction_key: str = ct.PREDICTION,
        target_key: str = ct.TARGET,
        mask_key: str = ct.MASK,
        weight: float = 1.0,
        valid_value: bool = True,
        mask_in_batch: bool = True,
    ) -> None:
        super().__init__()
        self.criterion = setup_module(criterion)
        self._prediction_key = prediction_key
        self._target_key = target_key
        self._mask_key = mask_key
        self._weight = float(weight)
        self._valid_value = bool(valid_value)
        self._mask_in_batch = bool(mask_in_batch)

    def extra_repr(self) -> str:
        return str_mapping(
            {
                "prediction": self._prediction_key,
                "target": self._target_key,
                "mask": self._mask_key,
                "weight": self._weight,
            }
        )

    def forward(self, net_out: dict, batch: dict) -> dict[str, Tensor]:
        r"""Return the loss value given the network output and the batch.

        Args:
            net_out: The network output which contains the prediction.
            batch: The batch which contains the target.

        Returns:
            A dict with the loss value.
        """
        # See the batch of sequences as a batch of examples
        prediction = net_out[self._prediction_key].flatten(0, 1)
        target = batch[self._target_key].flatten(0, 1)

        # Get the mask and remove the examples that are masked
        mask = (
            batch.get(self._mask_key, None)
            if self._mask_in_batch
            else net_out.get(self._mask_key, None)
        )
        prediction, target = self._mask_inputs(prediction=prediction, target=target, mask=mask)
        return {ct.LOSS: self.criterion(prediction, target)}

    def _mask_inputs(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            return prediction, target
        mask = mask.flatten().bool()
        if not self._valid_value:
            mask = torch.logical_not(mask)
        return prediction[mask], target[mask]
