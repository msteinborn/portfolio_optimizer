"""Unit tests for default dataset and stdlib CSV import/export helpers."""

import csv
import io

import numpy as np
import pytest

from src.data import (
    export_weights_csv,
    get_default_preset,
    load_market_assumptions_csv,
    parse_market_assumptions_csv_text,
    save_market_assumptions_csv,
    serialize_market_assumptions_csv,
    serialize_weights_csv,
)
from src.optimizer import (
    build_covariance_matrix,
    is_positive_semidefinite,
    validate_correlation_matrix,
)


def test_default_preset_is_valid_and_psd():
    preset = get_default_preset()
    assert len(preset.names) == 5
    assert preset.expected_returns.shape == (5,)
    assert preset.volatilities.shape == (5,)
    assert preset.variances.shape == (5,)
    assert preset.corr_matrix.shape == (5, 5)
    assert np.allclose(preset.variances, preset.volatilities**2)

    validated = validate_correlation_matrix(preset.corr_matrix)
    assert is_positive_semidefinite(validated)
    cov = build_covariance_matrix(validated, volatilities=preset.volatilities)
    assert is_positive_semidefinite(cov)


def test_csv_round_trip(tmp_path):
    preset = get_default_preset()
    csv_path = tmp_path / "markets.csv"

    save_market_assumptions_csv(preset, csv_path)
    loaded = load_market_assumptions_csv(csv_path, risk_free_rate=preset.risk_free_rate)

    assert loaded.names == preset.names
    assert np.allclose(loaded.expected_returns, preset.expected_returns)
    assert np.allclose(loaded.volatilities, preset.volatilities)
    assert np.allclose(loaded.variances, preset.variances)
    assert np.allclose(loaded.corr_matrix, preset.corr_matrix)
    assert loaded.risk_free_rate == preset.risk_free_rate


def test_csv_in_memory_round_trip():
    preset = get_default_preset()
    csv_text = serialize_market_assumptions_csv(preset)
    loaded = parse_market_assumptions_csv_text(
        csv_text, risk_free_rate=preset.risk_free_rate
    )
    assert loaded.names == preset.names
    assert np.allclose(loaded.expected_returns, preset.expected_returns)
    assert np.allclose(loaded.volatilities, preset.volatilities)
    assert np.allclose(loaded.corr_matrix, preset.corr_matrix)


def test_load_csv_with_variance_only(tmp_path):
    csv_path = tmp_path / "variance_only.csv"
    csv_path.write_text(
        "market,expected_return,variance,US,UK\n"
        "US,0.08,0.04,1.0,0.6\n"
        "UK,0.06,0.0225,0.6,1.0\n",
        encoding="utf-8",
    )

    loaded = load_market_assumptions_csv(csv_path)
    assert np.allclose(loaded.volatilities, [0.20, 0.15])
    assert np.allclose(loaded.variances, [0.04, 0.0225])


def test_load_csv_rejects_non_psd(tmp_path):
    csv_path = tmp_path / "bad_corr.csv"
    csv_path.write_text(
        "market,expected_return,volatility,A,B,C\n"
        "A,0.08,0.15,1.0,0.9,-0.9\n"
        "B,0.07,0.15,0.9,1.0,0.9\n"
        "C,0.06,0.15,-0.9,0.9,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive semi-definite"):
        load_market_assumptions_csv(csv_path)


def test_export_weights_csv(tmp_path):
    out_path = tmp_path / "weights.csv"
    export_weights_csv(["US", "UK"], np.array([0.65, 0.35]), out_path)

    rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
    assert [r["market"] for r in rows] == ["US", "UK"]
    assert np.allclose([float(r["weight"]) for r in rows], [0.65, 0.35])
    assert np.allclose([float(r["weight_pct"]) for r in rows], [65.0, 35.0])


def test_serialize_weights_csv():
    text = serialize_weights_csv(["US", "Risk-Free Asset"], np.array([0.8, 0.2]))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["market"] for r in rows] == ["US", "Risk-Free Asset"]
    assert np.allclose([float(r["weight_pct"]) for r in rows], [80.0, 20.0])
