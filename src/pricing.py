"""Pluggable asset pricing models that produce expected returns (mu).

Zero UI imports. The optimizer only needs mu and Sigma, so any model that
returns an expected-return vector can feed the Portfolio Optimizer tab.

Design notes:
- The base contract is `AssetPricingModel.price(inputs) -> PricingResult`.
  Linear factor models are one subclass (`LinearFactorModel`), not the base.
- `PricingResult.factor_contributions` is optional (None for non-additive models).
- Models are registered by key in `PRICING_MODELS` and looked up via
  `get_pricing_model` / `list_pricing_models`.

All rates are annualized decimals (0.045 = 4.5%).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class FactorSpec:
    """Declarative description of a single risk factor."""

    key: str
    label: str
    default_premium: float
    default_loading: float = 1.0


@dataclass(frozen=True)
class PricingInputs:
    """Inputs shared by all pricing models.

    - `factor_premia`: factor key -> annualized risk premium (decimal).
    - `loadings`: shape (N, K), one row per market, one column per factor
      (in the order of the model's `factors`). Ignored by models that do not
      use loadings.
    - `alphas`: shape (N,), annualized abnormal return per market (decimal).
      `None` means zeros.
    """

    risk_free_rate: float
    factor_premia: Mapping[str, float]
    loadings: Optional[np.ndarray] = None
    alphas: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PricingResult:
    """Output of a pricing model.

    `factor_contributions` maps factor key -> per-market contribution
    (loading * premium). It is None for models that are not additive.
    """

    expected_returns: np.ndarray
    risk_free_rate: float
    factor_contributions: Optional[dict[str, np.ndarray]] = None
    alpha_component: Optional[np.ndarray] = None


class AssetPricingModel(ABC):
    """Base class for all asset pricing models."""

    key: str = ""
    label: str = ""

    @property
    def factors(self) -> tuple[FactorSpec, ...]:
        """Factors this model exposes in the UI. Empty for models without factors."""
        return ()

    @abstractmethod
    def price(self, inputs: PricingInputs) -> PricingResult:
        """Compute expected returns for every market."""

    def expected_returns(self, inputs: PricingInputs) -> np.ndarray:
        """Convenience accessor returning only mu."""
        return self.price(inputs).expected_returns


@dataclass(frozen=True)
class LinearFactorModel(AssetPricingModel):
    """Linear factor model: E[R_i] = R_f + sum_k beta_{i,k} * lambda_k + alpha_i."""

    key: str = ""
    label: str = ""
    _factors: tuple[FactorSpec, ...] = field(default_factory=tuple)

    @property
    def factors(self) -> tuple[FactorSpec, ...]:
        return self._factors

    def price(self, inputs: PricingInputs) -> PricingResult:
        if not np.isfinite(inputs.risk_free_rate):
            raise ValueError("Risk-free rate must be finite.")

        k = len(self._factors)
        if k == 0:
            raise ValueError(f"Model '{self.label}' has no factors defined.")

        premia_vec = np.empty(k, dtype=float)
        for idx, spec in enumerate(self._factors):
            if spec.key not in inputs.factor_premia:
                raise ValueError(f"Missing risk premium for factor '{spec.label}'.")
            value = float(inputs.factor_premia[spec.key])
            if not np.isfinite(value):
                raise ValueError(f"Risk premium for '{spec.label}' must be finite.")
            premia_vec[idx] = value

        if inputs.loadings is None:
            raise ValueError("Factor loadings are required for a linear factor model.")
        loadings = np.asarray(inputs.loadings, dtype=float)
        if loadings.ndim == 1 and k == 1:
            loadings = loadings.reshape(-1, 1)
        if loadings.ndim != 2 or loadings.shape[1] != k:
            raise ValueError(
                f"Loadings must have shape (N, {k}); got {loadings.shape}."
            )
        if loadings.shape[0] == 0:
            raise ValueError("At least one market is required.")
        if not np.all(np.isfinite(loadings)):
            raise ValueError("Factor loadings must be finite.")

        n = loadings.shape[0]
        if inputs.alphas is None:
            alphas = np.zeros(n, dtype=float)
        else:
            alphas = np.asarray(inputs.alphas, dtype=float)
            if alphas.shape != (n,):
                raise ValueError(f"Alphas must have length {n}; got {alphas.shape}.")
            if not np.all(np.isfinite(alphas)):
                raise ValueError("Alphas must be finite.")

        contributions: dict[str, np.ndarray] = {}
        for idx, spec in enumerate(self._factors):
            contributions[spec.key] = loadings[:, idx] * premia_vec[idx]

        factor_sum = np.sum(np.column_stack(list(contributions.values())), axis=1)
        expected = float(inputs.risk_free_rate) + factor_sum + alphas

        return PricingResult(
            expected_returns=expected,
            risk_free_rate=float(inputs.risk_free_rate),
            factor_contributions=contributions,
            alpha_component=alphas,
        )


# ---------------------------------------------------------------------------
# Built-in models
# ---------------------------------------------------------------------------

GLOBAL_CAPM = LinearFactorModel(
    key="global_capm",
    label="Global CAPM",
    _factors=(
        FactorSpec(
            key="erp",
            label="Global Equity Risk Premium (ERP)",
            default_premium=0.045,
            default_loading=1.0,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRICING_MODELS: dict[str, AssetPricingModel] = {}


def register_pricing_model(model: AssetPricingModel) -> None:
    """Register a pricing model under its `key`. Re-registering a key replaces it."""
    if not model.key:
        raise ValueError("Pricing model must define a non-empty `key`.")
    PRICING_MODELS[model.key] = model


def get_pricing_model(key: str) -> AssetPricingModel:
    """Look up a registered pricing model by key."""
    try:
        return PRICING_MODELS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown pricing model '{key}'.") from exc


def list_pricing_models() -> list[AssetPricingModel]:
    """All registered pricing models, in registration order."""
    return list(PRICING_MODELS.values())


register_pricing_model(GLOBAL_CAPM)
