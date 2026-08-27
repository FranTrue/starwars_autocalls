"""Tests for data/integration.py, using small hand-made dataframes."""
import pandas as pd

from starwars_autocalls.data.integration import aggregate_basket_features, merge_trading_datasets


def _fake_rfqs() -> pd.DataFrame:
    return pd.DataFrame({
        "rfq_id": ["RFQ-001", "RFQ-002"],
        "underlyings": [["AAA"], ["AAA", "BBB"]],
        "requested_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
    })


def _fake_ref() -> pd.DataFrame:
    return pd.DataFrame({
        "underlying": ["AAA", "BBB"],
        "sector": ["Tech", "Energy"],
        "structural_base_vol": [0.20, 0.30],
    })


def _fake_vol() -> pd.DataFrame:
    return pd.DataFrame({
        "underlying": ["AAA", "BBB"],
        "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        "realized_vol_63d": [0.25, 0.45],
    })


def test_merge_trading_datasets_explodes_one_row_per_underlying():
    df_merged = merge_trading_datasets(df_rfqs=_fake_rfqs(), df_ref=_fake_ref(), df_vol=_fake_vol())

    assert len(df_merged) == 3
    assert set(df_merged.loc[df_merged["rfq_id"] == "RFQ-002", "underlying"]) == {"AAA", "BBB"}


def test_merge_trading_datasets_brings_in_the_right_volatility():
    df_merged = merge_trading_datasets(df_rfqs=_fake_rfqs(), df_ref=_fake_ref(), df_vol=_fake_vol())

    bbb_row = df_merged[df_merged["underlying"] == "BBB"].iloc[0]
    assert bbb_row["realized_vol_63d"] == 0.45
    assert bbb_row["structural_base_vol"] == 0.30


def test_aggregate_single_underlying_basket_has_zero_std():
    df_merged = merge_trading_datasets(df_rfqs=_fake_rfqs(), df_ref=_fake_ref(), df_vol=_fake_vol())
    df_agg = aggregate_basket_features(df_merged)

    single_basket = df_agg[df_agg["rfq_id"] == "RFQ-001"].iloc[0]
    assert single_basket["n_underlyings"] == 1
    assert single_basket["realized_vol_63d_std"] == 0.0


def test_aggregate_two_underlying_basket_max_is_the_worse_case():
    df_merged = merge_trading_datasets(df_rfqs=_fake_rfqs(), df_ref=_fake_ref(), df_vol=_fake_vol())
    df_agg = aggregate_basket_features(df_merged)

    worst_of_basket = df_agg[df_agg["rfq_id"] == "RFQ-002"].iloc[0]
    assert worst_of_basket["n_underlyings"] == 2
    assert worst_of_basket["realized_vol_63d_max"] == 0.45
    assert worst_of_basket["realized_vol_63d_min"] == 0.25
