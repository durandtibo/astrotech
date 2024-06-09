r"""Contain the record implementations."""

from __future__ import annotations

__all__ = ["BaseRecord", "get_max_size", "set_max_size"]

from astrotech.record._config import get_max_size, set_max_size
from astrotech.record.base import BaseRecord
