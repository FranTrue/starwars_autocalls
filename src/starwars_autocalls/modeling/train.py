"""Trains the duration-prediction model and saves it."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from starwars_autocalls.config import CATEGORICAL_COLUMNS, LEAKAGE_OR_ID_COLUMNS, MODEL_DIR
from starwars_autocalls.features.build_features import build_features


def load_training_data() -> pd.DataFrame:
    """Rebuilds the training table from the raw CSVs."""
    return build_features()


def split_features_and_target(df: pd.DataFrame):
    """Splits executed RFQs into features (X) and target (y)."""
    df_executed = df[df["executed"]].copy()

    y = df_executed["avg_duration_months"]
    X = df_executed.drop(columns=LEAKAGE_OR_ID_COLUMNS)

    return X, y


def train_test_split_by_date(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Splits chronologically: earliest rows train, most recent test."""
    sorted_index = X["requested_date"].sort_values().index
    split_point = int(len(sorted_index) * (1 - test_size))

    train_idx = sorted_index[:split_point]
    test_idx = sorted_index[split_point:]

    X_train = X.loc[train_idx].drop(columns=["requested_date"])
    X_test = X.loc[test_idx].drop(columns=["requested_date"])
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    return X_train, X_test, y_train, y_test


def _build_preprocessor() -> ColumnTransformer:
    """One-hot encodes categorical columns, passes numeric columns through."""
    return ColumnTransformer(
        transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)],
        remainder="passthrough",
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Returns the baseline and production model pipelines."""
    return {
        "baseline_linear_regression": Pipeline([
            ("preprocessor", _build_preprocessor()),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", _build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1,
            )),
        ]),
    }


def save_model(pipeline: Pipeline, filename: str = "model.joblib", compress: int = 3) -> Path:
    """Saves the fitted pipeline as a compressed artifact."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump(pipeline, path, compress=compress)
    return path


def train_production_model() -> Pipeline:
    """Fits and returns the random_forest model that gets shipped."""
    df = load_training_data()
    X, y = split_features_and_target(df)
    X_train, _X_test, y_train, _y_test = train_test_split_by_date(X, y)

    pipeline = build_pipelines()["random_forest"]
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == "__main__":
    trained_pipeline = train_production_model()
    saved_path = save_model(trained_pipeline)
    print(f"Saved trained model to {saved_path}")
