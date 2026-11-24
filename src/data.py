"""Default international equity dataset and stdlib CSV import/export helpers.

Uses Python's built-in `csv` module (no pandas) for fast Pyodide static web loading.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.optimizer import validate_correlation_matrix


@dataclass(frozen=True)
class MarketDataset:
    """Container for international equity market assumptions."""

    names: list[str]
    expected_returns: np.ndarray
    volatilities: np.ndarray
    corr_matrix: np.ndarray
    risk_free_rate: float = 0.03

    @property
    def variances(self) -> np.ndarray:
        """Annualized variances (sigma^2)."""
        return self.volatilities**2


def get_default_preset() -> MarketDataset:
    """Return a realistic 5-market international equity preset.

    Markets: US, Europe ex-UK, Japan, UK, Emerging Markets.
    Returns and volatilities are annualized decimals (e.g., 0.085 = 8.5%).
    """
    names = ["US", "Europe ex-UK", "Japan", "UK", "Emerging Markets"]
    expected_returns = np.array([0.085, 0.072, 0.060, 0.068, 0.095], dtype=float)
    volatilities = np.array([0.16, 0.18, 0.17, 0.15, 0.23], dtype=float)
    corr_matrix = np.array(
        [
            [1.00, 0.78, 0.55, 0.72, 0.65],
            [0.78, 1.00, 0.60, 0.85, 0.70],
            [0.55, 0.60, 1.00, 0.58, 0.52],
            [0.72, 0.85, 0.58, 1.00, 0.66],
            [0.65, 0.70, 0.52, 0.66, 1.00],
        ],
        dtype=float,
    )
    corr_matrix = validate_correlation_matrix(corr_matrix)
    return MarketDataset(
        names=names,
        expected_returns=expected_returns,
        volatilities=volatilities,
        corr_matrix=corr_matrix,
        risk_free_rate=0.03,
    )


def serialize_market_assumptions_csv(dataset: MarketDataset) -> str:
    """Serialize market assumptions to a CSV string."""
    n = len(dataset.names)
    if (
        len(dataset.expected_returns) != n
        or len(dataset.volatilities) != n
        or dataset.corr_matrix.shape != (n, n)
    ):
        raise ValueError("Dataset dimensions are inconsistent.")

    corr = validate_correlation_matrix(dataset.corr_matrix)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["market", "expected_return", "volatility", "variance", *dataset.names])
    for i, name in enumerate(dataset.names):
        row = [
            name,
            f"{float(dataset.expected_returns[i]):.10g}",
            f"{float(dataset.volatilities[i]):.10g}",
            f"{float(dataset.variances[i]):.10g}",
            *(f"{float(corr[i, j]):.10g}" for j in range(n)),
        ]
        writer.writerow(row)
    return buf.getvalue()


def save_market_assumptions_csv(
    dataset: MarketDataset, filepath: str | Path
) -> None:
    """Save market assumptions (returns, volatilities, variances, correlations) to CSV."""
    csv_text = serialize_market_assumptions_csv(dataset)
    Path(filepath).write_text(csv_text, encoding="utf-8")


def parse_market_assumptions_csv_text(
    csv_text: str, risk_free_rate: float = 0.03
) -> MarketDataset:
    """Parse market assumptions from a CSV string and validate the correlation matrix.

    Expected columns:
    - `market`: market name
    - `expected_return`: annualized expected return (decimal)
    - `volatility` and/or `variance`: annualized volatility or variance (decimal)
    - One correlation column per market name matching the `market` rows
    """
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    fieldnames = [c.strip() for c in (reader.fieldnames or [])]
    rows = [
        {k.strip(): (v.strip() if v is not None else "") for k, v in row.items() if k is not None}
        for row in reader
    ]
    if not rows:
        raise ValueError("CSV file is empty.")

    if "market" not in fieldnames or "expected_return" not in fieldnames:
        raise ValueError("CSV must contain 'market' and 'expected_return' columns.")

    if "volatility" not in fieldnames and "variance" not in fieldnames:
        raise ValueError("CSV must contain either a 'volatility' or 'variance' column.")

    names = [row["market"] for row in rows]
    if any(not name for name in names):
        raise ValueError("Market names cannot be empty.")
    if len(set(names)) != len(names):
        raise ValueError("Market names must be unique.")

    try:
        expected_returns = np.array(
            [float(row["expected_return"]) for row in rows], dtype=float
        )
    except ValueError as exc:
        raise ValueError("Expected returns must be numeric.") from exc

    if "volatility" in fieldnames:
        try:
            volatilities = np.array(
                [float(row["volatility"]) for row in rows], dtype=float
            )
        except ValueError as exc:
            raise ValueError("Volatilities must be numeric.") from exc
        if np.any(volatilities < 0.0) or not np.all(np.isfinite(volatilities)):
            raise ValueError("Volatilities must be finite and non-negative.")
    else:
        try:
            variances = np.array(
                [float(row["variance"]) for row in rows], dtype=float
            )
        except ValueError as exc:
            raise ValueError("Variances must be numeric.") from exc
        if np.any(variances < 0.0) or not np.all(np.isfinite(variances)):
            raise ValueError("Variances must be finite and non-negative.")
        volatilities = np.sqrt(variances)

    missing_corr_cols = [name for name in names if name not in fieldnames]
    if missing_corr_cols:
        raise ValueError(
            f"CSV is missing correlation columns for markets: {', '.join(missing_corr_cols)}"
        )

    n = len(names)
    corr_matrix = np.zeros((n, n), dtype=float)
    try:
        for i, row in enumerate(rows):
            for j, col_name in enumerate(names):
                corr_matrix[i, j] = float(row[col_name])
    except ValueError as exc:
        raise ValueError("Correlation matrix values must be numeric.") from exc

    corr_matrix = validate_correlation_matrix(corr_matrix)

    return MarketDataset(
        names=names,
        expected_returns=expected_returns,
        volatilities=volatilities,
        corr_matrix=corr_matrix,
        risk_free_rate=float(risk_free_rate),
    )


def load_market_assumptions_csv(
    filepath: str | Path, risk_free_rate: float = 0.03
) -> MarketDataset:
    """Load market assumptions from a CSV file path."""
    csv_text = Path(filepath).read_text(encoding="utf-8")
    return parse_market_assumptions_csv_text(csv_text, risk_free_rate=risk_free_rate)


def serialize_weights_csv(names: Sequence[str], weights: np.ndarray) -> str:
    """Serialize optimal portfolio weights to a CSV string."""
    w = np.asarray(weights, dtype=float)
    if len(names) != len(w):
        raise ValueError("Number of market names must match number of weights.")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["market", "weight", "weight_pct"])
    for name, weight in zip(names, w):
        writer.writerow([name, f"{float(weight):.10g}", f"{float(weight * 100.0):.6f}"])
    return buf.getvalue()


def export_weights_csv(
    names: Sequence[str],
    weights: np.ndarray,
    filepath: str | Path,
) -> None:
    """Export optimal portfolio weights to a CSV file path."""
    csv_text = serialize_weights_csv(names, weights)
    Path(filepath).write_text(csv_text, encoding="utf-8")
