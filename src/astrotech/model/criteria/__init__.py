r"""Contain criteria (a.k.a loss) modules."""

from __future__ import annotations

__all__ = [
    "Loss",
    "ModeLoss",
    "NoLoss",
    "PaddedSequenceLoss",
    "SumLoss",
]

from astrotech.model.criteria.mode import ModeLoss
from astrotech.model.criteria.no import NoLoss
from astrotech.model.criteria.padded_seq import PaddedSequenceLoss
from astrotech.model.criteria.sum import SumLoss
from astrotech.model.criteria.vanilla import Loss
