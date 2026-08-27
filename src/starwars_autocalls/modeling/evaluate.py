"""Scores a trained model and reports feature importances."""
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from starwars_autocalls.config import MODEL_DIR
from starwars_autocalls.modeling.train import (
    build_pipelines,
    load_training_data,
    split_features_and_target,
    train_test_split_by_date,
)


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes MAE, RMSE and R2 on held-out data."""
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mse ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }


def print_feature_importances(pipeline: Pipeline, top_n: int = 10) -> None:
    """Prints the top_n most influential features."""
    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    print(importances.sort_values(ascending=False).head(top_n))


if __name__ == "__main__":
    df = load_training_data()
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y)

    saved_pipeline = joblib.load(MODEL_DIR / "model.joblib")

    baseline_pipeline = build_pipelines()["baseline_linear_regression"]
    baseline_pipeline.fit(X_train, y_train)

    print("Model comparison (test set, held-out by date):")
    for name, pipeline in [("baseline_linear_regression", baseline_pipeline), ("random_forest (saved)", saved_pipeline)]:
        metrics = evaluate(pipeline, X_test, y_test)
        print(f"  {name}: MAE={metrics['mae']:.2f} months, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")

    print("\nTop feature importances (saved random_forest):")
    print_feature_importances(saved_pipeline)
