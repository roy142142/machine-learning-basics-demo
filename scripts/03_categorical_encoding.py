"""
Demo for handling categorical variables in tabular regression data.

Concepts demonstrated:
- Dropping categorical variables
- Label encoding
- One-hot encoding
- Low-cardinality vs high-cardinality columns
- MAE-based comparison
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


TARGET = "SalePrice"


def load_train_test_data(
    train_path: str | Path = "train.csv",
    test_path: str | Path = "test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files."""
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train.csv and test.csv in the repository root.")

    train_data = pd.read_csv(train_path, index_col="Id")
    test_data = pd.read_csv(test_path, index_col="Id")

    return train_data, test_data


def score_dataset(X_train, X_valid, y_train, y_valid) -> float:
    """Train Random Forest and return validation MAE."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, predictions)


def one_hot_encoder_compatible() -> OneHotEncoder:
    """
    Create OneHotEncoder compatible with newer and older scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main() -> None:
    train_data, _ = load_train_test_data()

    train_data.dropna(axis=0, subset=[TARGET], inplace=True)
    y = train_data[TARGET]
    X = train_data.drop([TARGET], axis=1)

    # Drop columns with missing values to keep this encoding demo simple.
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X = X.drop(cols_with_missing, axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )

    object_cols = [
        col for col in X_train.columns
        if X_train[col].dtype == "object"
    ]

    print("Categorical columns:")
    print(object_cols)

    # Approach 1: Drop categorical variables
    drop_X_train = X_train.select_dtypes(exclude=["object"])
    drop_X_valid = X_valid.select_dtypes(exclude=["object"])

    drop_mae = score_dataset(
        drop_X_train,
        drop_X_valid,
        y_train,
        y_valid,
    )

    print(f"\nMAE - Drop categorical variables: {drop_mae:.2f}")

    # Approach 2: Label encoding only for safe columns
    good_label_cols = [
        col for col in object_cols
        if set(X_train[col]) == set(X_valid[col])
    ]

    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    print("\nCategorical columns selected for label encoding:")
    print(good_label_cols)

    print("\nCategorical columns dropped before label encoding:")
    print(bad_label_cols)

    label_X_train = X_train.drop(bad_label_cols, axis=1).copy()
    label_X_valid = X_valid.drop(bad_label_cols, axis=1).copy()

    label_encoder = LabelEncoder()

    for col in good_label_cols:
        label_X_train[col] = label_encoder.fit_transform(label_X_train[col])
        label_X_valid[col] = label_encoder.transform(label_X_valid[col])

    label_mae = score_dataset(
        label_X_train,
        label_X_valid,
        y_train,
        y_valid,
    )

    print(f"\nMAE - Label encoding: {label_mae:.2f}")

    # Approach 3: One-hot encoding for low-cardinality columns
    cardinality = {
        col: X_train[col].nunique()
        for col in object_cols
    }

    print("\nCategorical cardinality:")
    print(sorted(cardinality.items(), key=lambda item: item[1]))

    low_cardinality_cols = [
        col for col in object_cols
        if X_train[col].nunique() < 10
    ]

    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    print("\nCategorical columns selected for one-hot encoding:")
    print(low_cardinality_cols)

    print("\nCategorical columns dropped due to high cardinality:")
    print(high_cardinality_cols)

    encoder = one_hot_encoder_compatible()

    OH_cols_train = pd.DataFrame(
        encoder.fit_transform(X_train[low_cardinality_cols]),
        index=X_train.index,
    )

    OH_cols_valid = pd.DataFrame(
        encoder.transform(X_valid[low_cardinality_cols]),
        index=X_valid.index,
    )

    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    OH_mae = score_dataset(
        OH_X_train,
        OH_X_valid,
        y_train,
        y_valid,
    )

    print(f"\nMAE - One-hot encoding: {OH_mae:.2f}")


if __name__ == "__main__":
    main()
