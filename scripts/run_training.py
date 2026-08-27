"""Runs the full pipeline: raw CSVs -> trained model.joblib."""
from starwars_autocalls.modeling.train import save_model, train_production_model

if __name__ == "__main__":
    print("Training model from the raw CSVs in data/raw/...")
    pipeline = train_production_model()
    saved_path = save_model(pipeline)
    print(f"Done. Model saved to {saved_path}")
