r"""Contain the models."""

from __future__ import annotations

__all__ = ["BaseModel", "Model", "is_model_config", "setup_model"]

from astrotech.model.base import BaseModel, is_model_config, setup_model
from astrotech.model.vanilla import Model
