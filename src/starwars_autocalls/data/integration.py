"""Combines the three raw sources into a single modeling-ready table."""
import pandas as pd
from starwars_autocalls.data.loaders import daily_volatility, rfqs, underlyings_reference


def merge_trading_datasets(df_rfqs=None, df_ref=None, df_vol=None) -> pd.DataFrame:
    """Merges RFQs, volatility and reference data into one row per underlying per RFQ."""
    df_rfqs = rfqs() if df_rfqs is None else df_rfqs
    df_ref = underlyings_reference() if df_ref is None else df_ref
    df_vol = daily_volatility() if df_vol is None else df_vol

    df_merged = df_rfqs.explode("underlyings").rename(columns={"underlyings": "underlying"})

    df_merged = pd.merge(df_merged, df_ref, on="underlying", how="left")

    df_merged = pd.merge(
        df_merged, df_vol,
        left_on=["underlying", "requested_date"], right_on=["underlying", "date"],
        how="left",
    )

    if "date" in df_merged.columns:
        df_merged = df_merged.drop(columns=["date"])

    return df_merged


def aggregate_basket_features(df_exploded: pd.DataFrame) -> pd.DataFrame:
    """Aggregates each basket's underlyings into one row per rfq_id."""
    agg_funcs = {
        "structural_base_vol": ["mean", "max", "min", "std"],
        "realized_vol_63d": ["mean", "max", "min", "std"],
        "underlying": "count",
    }
    df_agg = df_exploded.groupby("rfq_id").agg(agg_funcs)
    df_agg.columns = ["_".join(col).strip("_") for col in df_agg.columns]
    df_agg = df_agg.rename(columns={"underlying_count": "n_underlyings"})
    df_agg = df_agg.reset_index()

    std_cols = [col for col in df_agg.columns if col.endswith("_std")]
    df_agg[std_cols] = df_agg[std_cols].fillna(0)

    return df_agg


def combined() -> pd.DataFrame:
    """Returns the final modeling-ready table, one row per rfq_id."""
    df_exploded = merge_trading_datasets()
    df_agg = aggregate_basket_features(df_exploded)

    rfq_level_cols = [
        "rfq_id", "product_type", "basket_type", "autocall_barrier_pct",
        "protection_barrier_pct", "no_call_period_months",
        "observation_frequency_days", "quoted_implied_vol",
        "notional_credits", "counterparty", "trader_id", "requested_date",
        "executed", "start_date", "end_date", "avg_duration_months",
    ]
    df_rfq_level = df_exploded[rfq_level_cols].drop_duplicates(subset="rfq_id")

    return df_rfq_level.merge(df_agg, on="rfq_id", how="left")


if __name__ == "__main__":
    df_final = combined()
    print(df_final.head())
    print(df_final.shape)
