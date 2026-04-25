from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import DatasetId
from .datasets.registry import LoadedDataset, load_dataset
from .preprocessing import PreparedData, prepare_data, transform_X


@dataclass(frozen=True)
class NumpyData:
    prepared: PreparedData
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_and_prepare_numpy(
    *,
    dataset: DatasetId | str,
    path: str | Path,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[LoadedDataset, NumpyData]:
    ds = load_dataset(dataset, path)
    prepared = prepare_data(ds, seed=seed, test_size=test_size, val_size=val_size)

    splits = prepared.splits
    X_train = transform_X(prepared.preprocessor, splits.X_train)
    X_val = transform_X(prepared.preprocessor, splits.X_val)
    X_test = transform_X(prepared.preprocessor, splits.X_test)

    y_train = splits.y_train.to_numpy(dtype=np.int64)
    y_val = splits.y_val.to_numpy(dtype=np.int64)
    y_test = splits.y_test.to_numpy(dtype=np.int64)

    return ds, NumpyData(
        prepared=prepared,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
