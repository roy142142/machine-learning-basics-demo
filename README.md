# machine-learning-basics-demo
Educational machine learning basics demo using tabular regression data, including train-validation split, decision tree regression, random forest regression, and MAE-based model evaluation.
## Overview

This repository contains introductory machine learning practice scripts using tabular regression data. The project was developed as part of my early machine learning learning path and has been cleaned and reorganized for portfolio use.

The main objective is to demonstrate basic supervised learning workflows in Python using `pandas` and `scikit-learn`, including data preparation, model training, validation, preprocessing, categorical encoding, cross-validation, model comparison, and prediction file generation.

## Project Scope

This repository is not intended to present a production-grade machine learning system. Instead, it demonstrates core machine learning concepts using a structured and readable code format.

The examples focus on a house price regression task, where the goal is to predict a continuous target variable from tabular features.

## Key Concepts Demonstrated

- Loading tabular data with pandas
- Separating features and target variables
- Train-validation splitting
- Model training with Decision Tree and Random Forest regressors
- Model evaluation using Mean Absolute Error
- Handling missing values by dropping columns or imputing values
- Label encoding and one-hot encoding of categorical variables
- Handling low-cardinality and high-cardinality categorical features
- Building preprocessing and modeling pipelines
- Using cross-validation for more robust model evaluation
- Comparing multiple Random Forest configurations
- Generating prediction outputs for a test dataset
- Basic XGBoost regression workflow

## Repository Structure

```text
machine-learning-basics-demo/
│
├── README.md
├── requirements.txt
│
└── scripts/
    ├── 01_decision_tree_random_forest.py
    ├── 02_missing_value_handling.py
    ├── 03_categorical_encoding.py
    ├── 04_pipeline_cross_validation.py
    ├── 05_model_comparison_submission.py
    └── 06_xgboost_regression.py
