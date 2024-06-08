from __future__ import annotations

from unittest.mock import patch

import pytest

from astrotech.utils.time import sync_perf_counter, timeblock

#######################################
#     Tests for sync_perf_counter     #
#######################################


@patch("astrotech.utils.time.torch.cuda.is_available", lambda: False)
def test_sync_perf_counter_no_cuda() -> None:
    with patch("astrotech.utils.time.torch.cuda.synchronize") as synchronize_mock:
        assert isinstance(sync_perf_counter(), float)
        synchronize_mock.assert_not_called()


@patch("astrotech.utils.time.torch.cuda.is_available", lambda: True)
def test_sync_perf_counter_cuda() -> None:
    with patch("astrotech.utils.time.torch.cuda.synchronize") as synchronize_mock:
        assert isinstance(sync_perf_counter(), float)
        synchronize_mock.assert_called_once_with()


###############################
#     Tests for timeblock     #
###############################


def test_timeblock(caplog: pytest.LogCaptureFixture) -> None:
    with timeblock():
        pass  # do nothing
    assert len(caplog.messages) == 1
    assert caplog.messages[0].startswith("Total time: ")


def test_timeblock_custom_message(caplog: pytest.LogCaptureFixture) -> None:
    with timeblock("{time}"):
        pass  # do anything
    assert len(caplog.messages) == 1


def test_timeblock_custom_missing_time() -> None:
    with (
        pytest.raises(RuntimeError, match="{time} is missing in the message"),
        timeblock("message"),
    ):
        pass
