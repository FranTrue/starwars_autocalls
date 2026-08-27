"""Tests for features/build_features.py."""
from starwars_autocalls.features.build_features import build_features

EXPECTED_COLUMNS = {
    "rfq_id", "product_type", "basket_type", "executed", "avg_duration_months",
    "n_underlyings",
    "realized_vol_63d_mean", "realized_vol_63d_max",
    "realized_vol_63d_min", "realized_vol_63d_std",
}


def test_build_features_has_the_expected_columns():
    df = build_features()
    assert EXPECTED_COLUMNS.issubset(df.columns)


def test_build_features_has_one_row_per_rfq():
    df = build_features()
    assert df["rfq_id"].is_unique


def test_build_features_null_pattern_matches_executed():
    df = build_features()
    assert (df["avg_duration_months"].isna() == ~df["executed"]).all()
