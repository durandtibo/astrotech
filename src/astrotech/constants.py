r"""Contain values used across the package."""

from __future__ import annotations

__all__ = [
    "ARTIFACT_FOLDER_NAME",
    "CHECKPOINT",
    "CHECKPOINT_FOLDER_NAME",
    "DATA",
    "DATA_SOURCE",
    "EARLY_STOPPING",
    "ENGINE",
    "EVAL",
    "EVALUATION_LOOP",
    "EXP_TRACKER",
    "INPUT",
    "LOSS",
    "LR_SCHEDULER",
    "MASK",
    "MODEL",
    "OPTIMIZER",
    "OUTPUT",
    "PREDICTION",
    "RUNNER",
    "SCALER",
    "STATE",
    "TARGET",
    "TRAIN",
    "TRAINING_LOOP",
]

# These constants are used as default name for the training and evaluation metrics.
TRAIN = "train"
EVAL = "eval"

INPUT = "input"
LOSS = "loss"
MASK = "mask"
OUTPUT = "output"
PREDICTION = "prediction"
TARGET = "target"

# Engine keys
CHECKPOINT = "checkpoint"
DATA = "data"
DATA_SOURCE = "datasource"
ENGINE = "engine"
EVALUATION_LOOP = "evaluation_loop"
EXP_TRACKER = "exp_tracker"
LR_SCHEDULER = "lr_scheduler"
MODEL = "model"
OPTIMIZER = "optimizer"
RUNNER = "runner"
SCALER = "scaler"
STATE = "state"
TRAINING_LOOP = "training_loop"

EARLY_STOPPING = "early_stopping"

ARTIFACT_FOLDER_NAME = "artifacts"
CHECKPOINT_FOLDER_NAME = "checkpoints"
