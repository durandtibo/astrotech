from __future__ import annotations

import logging

import pytest

from astrotech.utils.logging import disable_logging

logger = logging.getLogger(__name__)


def log_something() -> None:
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")


#####################################
#     Tests for disable_logging     #
#####################################


@pytest.mark.parametrize(
    ("level", "num_lines"),
    [
        (logging.DEBUG - 1, 5),
        (logging.INFO - 1, 4),
        (logging.WARNING - 1, 3),
        (logging.ERROR - 1, 2),
        (logging.CRITICAL - 1, 1),
        ("DEBUG", 4),
        ("INFO", 3),
        ("WARNING", 2),
        ("ERROR", 1),
        ("CRITICAL", 0),
    ],
)
def test_disable_logging_level(
    caplog: pytest.LogCaptureFixture, level: int | str, num_lines: int
) -> None:
    with caplog.at_level(logging.NOTSET):
        with disable_logging(level):
            log_something()
        assert len(caplog.messages) == num_lines


@pytest.mark.parametrize(
    "level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
)
def test_disable_logging_reset_level_info(level: int | str) -> None:
    prev_level = logger.level
    with disable_logging(level):
        log_something()
    assert logger.level == prev_level
