"""Request and response schemas for the inference API."""
from pydantic import BaseModel


class RFQFeatures(BaseModel):
    """One RFQ's precomputed features, matching the model's training columns."""
    product_type: str
    basket_type: str
    autocall_barrier_pct: float
    protection_barrier_pct: float
    no_call_period_months: int
    observation_frequency_days: int
    quoted_implied_vol: float
    notional_credits: float
    counterparty: str
    trader_id: str
    structural_base_vol_mean: float
    structural_base_vol_max: float
    structural_base_vol_min: float
    structural_base_vol_std: float
    realized_vol_63d_mean: float
    realized_vol_63d_max: float
    realized_vol_63d_min: float
    realized_vol_63d_std: float
    n_underlyings: int


class DurationPrediction(BaseModel):
    """The model's predicted duration, in months."""
    predicted_avg_duration_months: float
