"""
Introductory regression demo using Decision Tree and Random Forest models.

Concepts demonstrated:
- Loading tabular data
- Selecting features and target
- Train-validation split
- Decision Tree regression
- Random Forest regression
- Mean Absolute Error evaluation
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


FEATURES = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
TARGET = "Price"


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load CSV data."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}. "
            "Place the dataset locally or update the file path."
        )

    return pd.read_csv(file_path)


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features and target variable."""
    required_columns = FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    return X, y


def evaluate_model(model, train_X, val_X, train_y, val_y) -> float:
    """Train model and return validation MAE."""
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)

    return mean_absolute_error(val_y, predictions)


def compare_tree_leaf_nodes(train_X, val_X, train_y, val_y) -> pd.DataFrame:
    """Compare Decision Tree models with different max_leaf_nodes."""
    results = []

    for max_leaf_nodes in [5, 50, 500, 5000]:
        model = DecisionTreeRegressor(
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
        )

        mae = evaluate_model(model, train_X, val_X, train_y, val_y)

        results.append(
            {
                "model": "DecisionTreeRegressor",
                "max_leaf_nodes": max_leaf_nodes,
                "validation_mae": round(mae, 2),
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    file_path = "melb_data.csv"
    data = load_data(file_path)

    X, y = prepare_features(data)

    print("Feature summary:")
    print(X.describe())

    print("\nTarget summary:")
    print(y.describe())

    train_X, val_X, train_y, val_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree_mae = evaluate_model(
        decision_tree,
        train_X,
        val_X,
        train_y,
        val_y,
    )

    print(f"\nDecision Tree validation MAE: {decision_tree_mae:.2f}")

    leaf_results = compare_tree_leaf_nodes(
        train_X,
        val_X,
        train_y,
        val_y,
    )

    print("\nDecision Tree max_leaf_nodes comparison:")
    print(leaf_results)

    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    random_forest_mae = evaluate_model(
        random_forest,
        train_X,
        val_X,
        train_y,
        val_y,
    )

    print(f"\nRandom Forest validation MAE: {random_forest_mae:.2f}")


if __name__ == "__main__":
    main()
