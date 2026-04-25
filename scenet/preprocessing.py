from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets.registry import LoadedDataset


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class PreparedData:
    splits: DataSplits
    preprocessor: ColumnTransformer
    feature_names: list[str]


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> DataSplits:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # val_size is fraction of total; convert to fraction of trainval
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac,
        random_state=seed,
        stratify=y_trainval,
    )

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def build_preprocessor(dataset: LoadedDataset) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    feature_name_combiner=lambda feature, category: f"{feature}={category}",
                ),
            ),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, dataset.numeric_cols),
            ("cat", cat_pipe, dataset.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def prepare_data(
    dataset: LoadedDataset,
    *,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> PreparedData:
    splits = make_splits(dataset.X, dataset.y, seed=seed, test_size=test_size, val_size=val_size)

    preprocessor = build_preprocessor(dataset)
    preprocessor.fit(splits.X_train)

    feature_names = list(preprocessor.get_feature_names_out())

    return PreparedData(
        splits=splits,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )


def transform_X(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    X_t = preprocessor.transform(X)
    # ColumnTransformer may return numpy array due to sparse_output=False
    return np.asarray(X_t, dtype=np.float32)
