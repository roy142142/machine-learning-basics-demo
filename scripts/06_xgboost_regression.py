"""
Introductory XGBoost regression demo.

Concepts demonstrated:
- Low-cardinality categorical feature selection
- Numerical feature selection
- One-hot encoding using pandas
- Aligning train/validation/test matrices
- XGBoost regression
- MAE evaluation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


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


def prepare_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare train/test features for XGBoost."""
    train_data = train_data.copy()

    train_data.dropna(axis=0, subset=[TARGET], inplace=True)

    y = train_data[TARGET]
    X = train_data.drop([TARGET], axis=1)

    low_cardinality_cols = [
        col for col in X.columns
        if X[col].nunique() < 10 and X[col].dtype == "object"
    ]

    numeric_cols = [
        col for col in X.columns
        if X[col].dtype in ["int64", "float64"]
    ]

    selected_cols = low_cardinality_cols + numeric_cols

    X_selected = X[selected_cols].copy()
    X_test_selected = test_data[selected_cols].copy()

    X_encoded = pd.get_dummies(X_selected)
    X_test_encoded = pd.get_dummies(X_test_selected)

    X_encoded, X_test_encoded = X_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0,
    )

    return X_encoded, y, X_test_encoded


def main() -> None:
    train_data, test_data = load_train_test_data()

    X, y, X_test = prepare_features(train_data, test_data)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=4,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, validation_predictions)

    print(f"XGBoost validation MAE: {mae:.2f}")


if __name__ == "__main__":
    main()
