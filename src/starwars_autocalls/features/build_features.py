"""Builds the modeling-ready feature table."""
import pandas as pd

from starwars_autocalls.config import PROCESSED_DATA_DIR
from starwars_autocalls.data.integration import combined


def build_features() -> pd.DataFrame:
    """Returns the modeling-ready table."""
    return combined()


if __name__ == "__main__":
    df_features = build_features()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "modeling_table.csv"
    df_features.to_csv(output_path, index=False)

    print(f"Saved {df_features.shape[0]} rows, {df_features.shape[1]} columns to {output_path}")
