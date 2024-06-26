from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from objectory import OBJECT_TARGET
from torch import Tensor
from torch.nn import CrossEntropyLoss, L1Loss, Module, MSELoss

from astrotech import constants as ct
from astrotech.model.criteria import PaddedSequenceLoss

SIZES = (1, 2, 3)


########################################
#     Tests for PaddedSequenceLoss     #
########################################


def test_padded_sequence_loss_repr() -> None:
    assert repr(PaddedSequenceLoss(torch.nn.MSELoss())).startswith("PaddedSequenceLoss(")


def test_padded_sequence_loss_str() -> None:
    assert str(PaddedSequenceLoss(torch.nn.MSELoss())).startswith("PaddedSequenceLoss(")


@pytest.mark.parametrize(
    ("criterion", "criterion_cls"),
    [
        (MSELoss(), MSELoss),
        (CrossEntropyLoss(), CrossEntropyLoss),
        ({OBJECT_TARGET: "torch.nn.MSELoss"}, MSELoss),
    ],
)
def test_padded_sequence_loss_criterion(
    criterion: dict | Module, criterion_cls: type[Module]
) -> None:
    assert isinstance(PaddedSequenceLoss(criterion).criterion, criterion_cls)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_3d_batch_first_reduction_average(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(batch_size, seq_len, feature_size, device=device)},
            {
                ct.TARGET: torch.ones(batch_size, seq_len, feature_size, device=device),
                ct.MASK: torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_3d_batch_first_reduction_none(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss(reduction="none")).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(batch_size, seq_len, feature_size, device=device)},
            {
                ct.TARGET: torch.ones(batch_size, seq_len, feature_size, device=device),
                ct.MASK: torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.zeros(seq_len * batch_size, feature_size, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_3d_sequence_first_reduction_average(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(seq_len, batch_size, feature_size, device=device)},
            {
                ct.TARGET: torch.ones(seq_len, batch_size, feature_size, device=device),
                ct.MASK: torch.ones(seq_len, batch_size, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_3d_sequence_first_reduction_none(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss(reduction="none")).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(seq_len, batch_size, feature_size, device=device)},
            {
                ct.TARGET: torch.ones(seq_len, batch_size, feature_size, device=device),
                ct.MASK: torch.ones(seq_len, batch_size, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.zeros(seq_len * batch_size, feature_size, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_4d_batch_first_reduction_average(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(batch_size, seq_len, feature_size, 4, device=device)},
            {
                ct.TARGET: torch.ones(batch_size, seq_len, feature_size, 4, device=device),
                ct.MASK: torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_4d_batch_first_reduction_none(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss(reduction="none")).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(batch_size, seq_len, feature_size, 4, device=device)},
            {
                ct.TARGET: torch.ones(batch_size, seq_len, feature_size, 4, device=device),
                ct.MASK: torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.zeros(seq_len * batch_size, feature_size, 4, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_4d_sequence_first_reduction_average(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(seq_len, batch_size, feature_size, 4, device=device)},
            {
                ct.TARGET: torch.ones(seq_len, batch_size, feature_size, 4, device=device),
                ct.MASK: torch.ones(seq_len, batch_size, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_mse_4d_sequence_first_reduction_none(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss(reduction="none")).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(seq_len, batch_size, feature_size, 4, device=device)},
            {
                ct.TARGET: torch.ones(seq_len, batch_size, feature_size, 4, device=device),
                ct.MASK: torch.ones(seq_len, batch_size, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.zeros(seq_len * batch_size, feature_size, 4, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_mse_mask_all_valid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, 4, device=device)},
            {
                ct.TARGET: torch.ones(2, 3, 4, device=device),
                ct.MASK: torch.ones(2, 3, 1, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_mse_mask_1_invalid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1], [4]], [[2], [5]], [[3], [6]]], dtype=torch.float, device=device
                )
            },
            {
                ct.TARGET: torch.tensor(
                    [[[2], [4]], [[1], [3]], [[2], [5]]], dtype=torch.float, device=device
                ),
                ct.MASK: torch.tensor([[1, 1], [1, 1], [1, 0]], device=device),
            },
        ),
        {ct.LOSS: torch.tensor(1.4, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_mse_mask_2_invalid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1], [4]], [[2], [5]], [[3], [6]]], dtype=torch.float, device=device
                )
            },
            {
                ct.TARGET: torch.tensor(
                    [[[2], [4]], [[1], [3]], [[2], [5]]], dtype=torch.float, device=device
                ),
                ct.MASK: torch.tensor([[1, 1], [1, 0], [1, 0]], device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.75, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_mse_mask_4_invalid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1], [4]], [[2], [5]], [[3], [6]]], dtype=torch.float, device=device
                )
            },
            {
                ct.TARGET: torch.tensor(
                    [[[2], [4]], [[1], [3]], [[2], [5]]], dtype=torch.float, device=device
                ),
                ct.MASK: torch.tensor([[1, 0], [1, 0], [0, 0]], device=device),
            },
        ),
        {ct.LOSS: torch.tensor(1.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_mse_2d_mask_2_invalid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float, device=device)},
            {
                ct.TARGET: torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float, device=device),
                ct.MASK: torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(3.75, dtype=torch.float, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_mse_2d_mask_4_invalid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float, device=device)},
            {
                ct.TARGET: torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float, device=device),
                ct.MASK: torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(2.5, dtype=torch.float, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_padded_sequence_loss_cross_entropy(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(CrossEntropyLoss()).to(device=device)
    loss = criterion(
        {ct.PREDICTION: torch.ones(batch_size, seq_len, feature_size, device=device)},
        {
            ct.TARGET: torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            ct.MASK: torch.ones(batch_size, seq_len, device=device),
        },
    )
    assert len(loss) == 1
    assert torch.is_tensor(loss[ct.LOSS])
    assert loss[ct.LOSS].ndim == 0
    assert loss[ct.LOSS].dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_cross_entropy_mask_all_valid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1, 2, 3], [2, 1, 3]], [[2, 1, 3], [1, 2, 3]]],
                    dtype=torch.float,
                    device=device,
                )
            },
            {
                ct.TARGET: torch.tensor([[1, 2], [1, 2]], dtype=torch.long, device=device),
                ct.MASK: torch.ones(2, 2, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(1.1576059644443804, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_cross_entropy_mask_1_invalid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1, 2, 3], [2, 1, 3]], [[2, 1, 3], [1, 2, 3]]],
                    dtype=torch.float,
                    device=device,
                )
            },
            {
                ct.TARGET: torch.tensor([[1, 2], [1, 2]], dtype=torch.long, device=device),
                ct.MASK: torch.tensor([[1, 1], [1, 0]], device=device),
            },
        ),
        {ct.LOSS: torch.tensor(1.4076059644443804, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_cross_entropy_mask_2_invalid(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1, 2, 3], [2, 1, 3]], [[2, 1, 3], [1, 2, 3]]],
                    dtype=torch.float,
                    device=device,
                )
            },
            {
                ct.TARGET: torch.tensor([[1, 2], [1, 2]], dtype=torch.long, device=device),
                ct.MASK: torch.tensor([[1, 1], [0, 0]], device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.9076059644443802, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_no_mask(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, 4, device=device)},
            {ct.TARGET: torch.ones(2, 3, 4, device=device)},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, 0], [True, 0], [0, 0]], dtype=torch.bool),
        torch.tensor([[1, 0], [1, 0], [0, 0]], dtype=torch.int),
        torch.tensor([[1, 0], [1, 0], [0, 0]], dtype=torch.long),
        torch.tensor([[1, 0], [1, 0], [0, 0]], dtype=torch.float),
    ],
)
def test_padded_sequence_loss_mask_dtype(device: str, mask: Tensor) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1], [4]], [[2], [5]], [[3], [6]]], dtype=torch.float, device=device
                )
            },
            {
                ct.TARGET: torch.tensor(
                    [[[2], [4]], [[1], [3]], [[2], [5]]], dtype=torch.float, device=device
                ),
                ct.MASK: mask.to(device=device),
            },
        ),
        {ct.LOSS: torch.tensor(1.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_forward_l1_correct_with_mask_true_bool(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(L1Loss(reduction="sum")).to(device=device)
    assert objects_are_equal(
        criterion(
            net_out={ct.PREDICTION: torch.zeros(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[True, True, False, True], [True, True, True, False]],
                    dtype=torch.bool,
                    device=device,
                ),
            },
        ),
        {ct.LOSS: torch.tensor(6.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_forward_l1_correct_with_mask_true_long(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(L1Loss(reduction="sum")).to(device=device)
    assert objects_are_equal(
        criterion(
            net_out={ct.PREDICTION: torch.zeros(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[1, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.long, device=device
                ),
            },
        ),
        {ct.LOSS: torch.tensor(6.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_forward_l1_correct_with_mask_false_bool(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(L1Loss(reduction="sum"), valid_value=False).to(device=device)
    assert objects_are_equal(
        criterion(
            net_out={ct.PREDICTION: torch.zeros(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[True, True, False, True], [True, True, True, False]],
                    dtype=torch.bool,
                    device=device,
                ),
            },
        ),
        {ct.LOSS: torch.tensor(2.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_forward_l1_correct_with_mask_false_long(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(L1Loss(reduction="sum"), valid_value=False).to(device=device)
    assert objects_are_equal(
        criterion(
            net_out={ct.PREDICTION: torch.zeros(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[1, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.long, device=device
                ),
            },
        ),
        {ct.LOSS: torch.tensor(2.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_padded_sequence_loss_mse_mask_in_batch_false(device: str) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(MSELoss(), mask_in_batch=False).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: torch.tensor(
                    [[[1], [4]], [[2], [5]], [[3], [6]]], dtype=torch.float, device=device
                ),
                ct.MASK: torch.tensor([[1, 1], [1, 1], [1, 0]], device=device),
            },
            {
                ct.TARGET: torch.tensor(
                    [[[2], [4]], [[1], [3]], [[2], [5]]], dtype=torch.float, device=device
                )
            },
        ),
        {ct.LOSS: torch.tensor(1.4, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prediction_key", ["my_prediction", "output"])
@pytest.mark.parametrize("target_key", ["my_target", "target"])
@pytest.mark.parametrize("mask_key", ["my_mask", "mask"])
def test_padded_sequence_loss_custom_keys(
    device: str, prediction_key: str, target_key: str, mask_key: str
) -> None:
    device = torch.device(device)
    criterion = PaddedSequenceLoss(
        MSELoss(),
        prediction_key=prediction_key,
        target_key=target_key,
        mask_key=mask_key,
    ).to(device=device)
    assert objects_are_equal(
        criterion(
            {prediction_key: torch.ones(2, 3, 4, device=device)},
            {
                target_key: torch.ones(2, 3, 4, device=device),
                mask_key: torch.ones(2, 3, dtype=torch.long, device=device),
            },
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )
