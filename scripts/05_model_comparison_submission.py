"""
Demo for comparing Random Forest configurations and generating a
submission-style prediction CSV.

Concepts demonstrated:
- Manual feature selection
- Model comparison
- Validation MAE
- Test prediction
- CSV output generation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


TARGET = "SalePrice"

FEATURES = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "MSSubClass",
    "OverallQual",
    "OverallCond",
    "YearRemodAdd",
    "KitchenAbvGr",
    "GrLivArea",
    "Fireplaces",
    "YrSold",
]


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


def score_model(model, X_train, X_valid, y_train, y_valid) -> float:
    """Train model and return validation MAE."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, predictions)


def get_models() -> list[RandomForestRegressor]:
    """Define Random Forest models for comparison."""
    return [
        RandomForestRegressor(n_estimators=50, random_state=42),
        RandomForestRegressor(n_estimators=100, random_state=42),
        RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=42),
        RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42),
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    ]


def main() -> None:
    train_data, test_data = load_train_test_data()

    train_data.dropna(axis=0, subset=[TARGET], inplace=True)

    y = train_data[TARGET]

    missing_features = [col for col in FEATURES if col not in train_data.columns]

    if missing_features:
        raise ValueError(f"Missing selected features in training data: {missing_features}")

    X = train_data[FEATURES].copy()
    X_test = test_data[FEATURES].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )

    models = get_models()

    model_results = []

    for index, model in enumerate(models, start=1):
        mae = score_model(model, X_train, X_valid, y_train, y_valid)
        model_results.append((index, model, mae))
        print(f"Model {index} validation MAE: {mae:.2f}")

    best_index, best_model, best_mae = min(
        model_results,
        key=lambda item: item[2],
    )

    print(f"\nBest model: Model {best_index} with MAE {best_mae:.2f}")

    # Refit best model on all selected training data before predicting test set.
    best_model.fit(X, y)
    test_predictions = best_model.predict(X_test)

    output = pd.DataFrame(
        {
            "Id": X_test.index,
            "SalePrice": test_predictions,
        }
    )

    output_file = Path("submission.csv")
    output.to_csv(output_file, index=False)

    print(f"Submission-style predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
