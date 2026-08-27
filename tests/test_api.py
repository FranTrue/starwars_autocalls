"""Tests for the inference API."""
from fastapi.testclient import TestClient

from starwars_autocalls.api.main import app

client = TestClient(app)


def _sample_payload() -> dict:
    """A valid RFQ payload matching RFQFeatures."""
    return {
        "product_type": "Snowball",
        "basket_type": "single",
        "autocall_barrier_pct": 1.0,
        "protection_barrier_pct": 0.7,
        "no_call_period_months": 6,
        "observation_frequency_days": 90,
        "quoted_implied_vol": 0.25,
        "notional_credits": 100000,
        "counterparty": "Jabba Asset Management",
        "trader_id": "TRD-001",
        "structural_base_vol_mean": 0.25,
        "structural_base_vol_max": 0.25,
        "structural_base_vol_min": 0.25,
        "structural_base_vol_std": 0.0,
        "realized_vol_63d_mean": 0.25,
        "realized_vol_63d_max": 0.25,
        "realized_vol_63d_min": 0.25,
        "realized_vol_63d_std": 0.0,
        "n_underlyings": 1,
    }


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_number():
    response = client.post("/predict", json=_sample_payload())

    assert response.status_code == 200
    body = response.json()
    assert "predicted_avg_duration_months" in body
    assert isinstance(body["predicted_avg_duration_months"], float)


def test_predict_missing_field_returns_422():
    """Missing a required field should return 422, not crash."""
    incomplete_payload = _sample_payload()
    del incomplete_payload["n_underlyings"]

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422
