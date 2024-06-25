r"""Contain criteria (a.k.a loss) modules."""

from __future__ import annotations

__all__ = ["Loss", "SumLoss"]

from astrotech.model.criteria.sum import SumLoss
from astrotech.model.criteria.vanilla import Loss
