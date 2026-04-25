from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TorchLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def make_loaders(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
) -> TorchLoaders:
    def ds(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        X_t = torch.as_tensor(X, dtype=torch.float32)
        # Ensure writable/copy to avoid undefined behavior warnings.
        y_np = np.asarray(y, dtype=np.float32).copy()
        y_t = torch.from_numpy(y_np)
        return TensorDataset(X_t, y_t)

    train_loader = DataLoader(
        ds(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        ds(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        ds(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return TorchLoaders(train=train_loader, val=val_loader, test=test_loader)
