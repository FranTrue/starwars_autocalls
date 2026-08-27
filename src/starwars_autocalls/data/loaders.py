"""Loads and validates the three raw CSV sources."""
import pandas as pd

from starwars_autocalls.config import OBSERVATION_FREQUENCY_DAYS, RAW_DATA_DIR


def _read_raw(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / filename)


def daily_volatility() -> pd.DataFrame:
    """Loads daily realized volatility per underlying."""
    df_vol_clean = _read_raw("daily_volatility.csv")

    if df_vol_clean.isnull().sum().sum() != 0:
        raise ValueError("unexpected nulls in daily_volatility")
    if df_vol_clean.duplicated(subset=["date", "underlying"]).any():
        raise ValueError("duplicate (date, underlying) rows")

    df_vol_clean["date"] = pd.to_datetime(df_vol_clean["date"])
    df_vol_clean = df_vol_clean.sort_values(["underlying", "date"]).reset_index(drop=True)

    return df_vol_clean


def rfqs() -> pd.DataFrame:
    """Loads and cleans the RFQ history."""
    df_rfqs_clean = _read_raw("rfqs.csv")

    df_rfqs_clean["underlyings"] = df_rfqs_clean["underlyings"].str.split("|")

    if (df_rfqs_clean["avg_duration_months"].isna() != ~df_rfqs_clean["executed"]).any():
        raise ValueError("avg_duration_months null pattern does not match executed")
    if df_rfqs_clean.duplicated(subset=["rfq_id"]).any():
        raise ValueError("duplicate rfq_id rows")

    df_rfqs_clean["requested_date"] = pd.to_datetime(df_rfqs_clean["requested_date"])
    df_rfqs_clean["start_date"] = pd.to_datetime(df_rfqs_clean["start_date"])
    df_rfqs_clean["end_date"] = pd.to_datetime(df_rfqs_clean["end_date"])

    df_rfqs_clean = df_rfqs_clean.sort_values(["requested_date", "rfq_id"]).reset_index(drop=True)

    not_executed = ~df_rfqs_clean["executed"]
    df_rfqs_clean.loc[not_executed, ["start_date", "end_date"]] = pd.NaT

    df_rfqs_clean["observation_frequency"] = (
        df_rfqs_clean["observation_frequency"].astype(str).str.strip().map(OBSERVATION_FREQUENCY_DAYS)
    )
    if df_rfqs_clean["observation_frequency"].isnull().any():
        raise ValueError("unmapped observation_frequency values produced nulls")

    df_rfqs_clean = df_rfqs_clean.rename(columns={"observation_frequency": "observation_frequency_days"})

    return df_rfqs_clean


def underlyings_reference() -> pd.DataFrame:
    """Loads the static per-ticker sector and volatility reference."""
    df_ref_clean = _read_raw("underlyings_reference.csv")

    if df_ref_clean.duplicated(subset=["underlying"]).any():
        raise ValueError("duplicate underlying rows")
    if not df_ref_clean["structural_base_vol"].between(0, 1).all():
        raise ValueError("structural_base_vol out of [0, 1] range")

    df_ref_clean = df_ref_clean.sort_values("underlying").reset_index(drop=True)

    return df_ref_clean


if __name__ == "__main__":
    print(daily_volatility().head())
    print(rfqs().head())
    print(underlyings_reference().head())
