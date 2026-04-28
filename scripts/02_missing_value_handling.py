"""
Demo for handling missing values in tabular regression data.

Concepts demonstrated:
- Dropping columns with missing values
- Median imputation
- Missing-value indicator columns
- Random Forest evaluation with MAE
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, predictions)


def add_missing_indicators(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    columns_with_missing: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add Boolean missing-value indicator columns."""
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    for col in columns_with_missing:
        X_train_plus[f"{col}_was_missing"] = X_train_plus[col].isnull()
        X_valid_plus[f"{col}_was_missing"] = X_valid_plus[col].isnull()

    return X_train_plus, X_valid_plus


def main() -> None:
    train_data, _ = load_train_test_data()

    train_data.dropna(axis=0, subset=[TARGET], inplace=True)
    y = train_data[TARGET]
    X = train_data.drop([TARGET], axis=1)

    X = X.select_dtypes(exclude=["object"])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )

    columns_with_missing = [
        col for col in X_train.columns
        if X_train[col].isnull().any()
    ]

    print("Columns with missing values:")
    print(columns_with_missing)

    # Approach 1: Drop columns with missing values
    reduced_X_train = X_train.drop(columns_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(columns_with_missing, axis=1)

    drop_mae = score_dataset(
        reduced_X_train,
        reduced_X_valid,
        y_train,
        y_valid,
    )

    print(f"\nMAE - Drop columns with missing values: {drop_mae:.2f}")

    # Approach 2: Median imputation
    median_imputer = SimpleImputer(strategy="median")

    imputed_X_train = pd.DataFrame(
        median_imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    imputed_X_valid = pd.DataFrame(
        median_imputer.transform(X_valid),
        columns=X_valid.columns,
        index=X_valid.index,
    )

    impute_mae = score_dataset(
        imputed_X_train,
        imputed_X_valid,
        y_train,
        y_valid,
    )

    print(f"MAE - Median imputation: {impute_mae:.2f}")

    # Approach 3: Imputation + missing indicators
    X_train_plus, X_valid_plus = add_missing_indicators(
        X_train,
        X_valid,
        columns_with_missing,
    )

    indicator_imputer = SimpleImputer(strategy="median")

    imputed_X_train_plus = pd.DataFrame(
        indicator_imputer.fit_transform(X_train_plus),
        columns=X_train_plus.columns,
        index=X_train_plus.index,
    )

    imputed_X_valid_plus = pd.DataFrame(
        indicator_imputer.transform(X_valid_plus),
        columns=X_valid_plus.columns,
        index=X_valid_plus.index,
    )

    indicator_mae = score_dataset(
        imputed_X_train_plus,
        imputed_X_valid_plus,
        y_train,
        y_valid,
    )

    print(f"MAE - Imputation with missing indicators: {indicator_mae:.2f}")


if __name__ == "__main__":
    main()
