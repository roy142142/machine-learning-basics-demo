"""
Demo for preprocessing pipelines and cross-validation.

Concepts demonstrated:
- SimpleImputer
- Pipeline
- ColumnTransformer
- Cross-validation with negative MAE scoring
- Preventing train-validation contamination by fitting preprocessing inside a pipeline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "SalePrice"


def load_data(train_path: str | Path = "train.csv") -> pd.DataFrame:
    """Load train CSV file."""
    train_path = Path(train_path)

    if not train_path.exists():
        raise FileNotFoundError("Expected train.csv in the repository root.")

    return pd.read_csv(train_path, index_col="Id")


def one_hot_encoder_compatible() -> OneHotEncoder:
    """Create OneHotEncoder compatible with different scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Build preprocessing + Random Forest pipeline."""
    numerical_transformer = SimpleImputer(strategy="mean")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder_compatible()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main() -> None:
    data = load_data()

    data.dropna(axis=0, subset=[TARGET], inplace=True)
    y = data[TARGET]
    X = data.drop([TARGET], axis=1)

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )

    categorical_cols = [
        col for col in X_train_full.columns
        if X_train_full[col].nunique() < 10 and X_train_full[col].dtype == "object"
    ]

    numerical_cols = [
        col for col in X_train_full.columns
        if X_train_full[col].dtype in ["int64", "float64"]
    ]

    selected_cols = categorical_cols + numerical_cols

    X_selected = X[selected_cols].copy()

    pipeline = build_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )

    scores = -1 * cross_val_score(
        pipeline,
        X_selected,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
    )

    print("MAE scores across folds:")
    print(scores)

    print(f"\nAverage MAE across folds: {scores.mean():.2f}")


if __name__ == "__main__":
    main()
