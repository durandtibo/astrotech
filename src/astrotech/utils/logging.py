r"""Contain utility functions to manage loggers."""

from __future__ import annotations

__all__ = ["disable_logging"]

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def disable_logging(level: int | str = logging.CRITICAL) -> Generator[None, None, None]:
    r"""Context manager to temporarily disable the logging.

    All logging calls of severity ``level`` and below will be
    disabled.

    Args:
        level: The logging level.

    Example usage:

    ```pycon

    >>> import logging
    >>> from astrotech.utils.logging import disable_logging
    >>> with disable_logging("INFO"):
    ...     logging.critical("CRITICAL")
    ...     logging.info("INFO")
    ...     logging.debug("DEBUG")
    ...
    CRITICAL:root:CRITICAL

    ```
    """
    prev_level = logging.getLogger(__name__).level
    if isinstance(level, str):
        level = logging.getLevelName(level)
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(prev_level)
