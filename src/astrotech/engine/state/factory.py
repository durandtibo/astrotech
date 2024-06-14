r"""Contain some utility functions for the engine state."""

from __future__ import annotations

__all__ = ["setup_engine_state"]

import logging

from astrotech.engine.state.base import BaseEngineState
from astrotech.engine.state.vanilla import EngineState
from astrotech.utils.factory import str_target_object

logger = logging.getLogger(__name__)


def setup_engine_state(state: BaseEngineState | dict | None) -> BaseEngineState:
    r"""Set up the engine state.

    The state is instantiated from its configuration by using the
    ``BaseEngineState`` factory function.

    Args:
        state: The engine state or its configuration. If ``None``, the
            ``EngineState`` is instantiated.

    Returns:
        The instantiated engine state.

    Example usage:

    ```pycon

    >>> from astrotech.engine.state import setup_engine_state
    >>> state = setup_engine_state({"_target_": "astrotech.engine.state.EngineState"})
    >>> state
    EngineState(
      (epoch): -1
      (iteration): -1
      (max_epochs): 1
      (modules): AssetManager(num_assets=0)
      (random_seed): 9984043075503325450
      (records): RecordManager()
    )

    ```
    """
    if state is None:
        state = EngineState()
    if isinstance(state, dict):
        logger.info(
            f"Initializing an engine state from its configuration... {str_target_object(state)}"
        )
        state = BaseEngineState.factory(**state)
    return state
