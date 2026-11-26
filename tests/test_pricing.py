"""Asset pricing tests: Global CAPM, pluggable model abstraction, registry, validation.

Run just these with:  .venv/bin/pytest tests/test_pricing.py -v
"""

import numpy as np
import pytest

from src.data import DEFAULT_BETA_BY_MARKET, DEFAULT_ERP, get_default_preset
from src.pricing import (
    GLOBAL_CAPM,
    PRICING_MODELS,
    AssetPricingModel,
    FactorSpec,
    LinearFactorModel,
    PricingInputs,
    PricingResult,
    get_pricing_model,
    list_pricing_models,
    register_pricing_model,
)

RF = 0.03
ERP = 0.045


def capm_inputs(betas, alphas=None, rf=RF, erp=ERP):
    return PricingInputs(
        risk_free_rate=rf,
        factor_premia={"erp": erp},
        loadings=np.asarray(betas, dtype=float).reshape(-1, 1),
        alphas=None if alphas is None else np.asarray(alphas, dtype=float),
    )


# ---------------------------------------------------------------------------
# Global CAPM math
# ---------------------------------------------------------------------------


class TestGlobalCAPM:
    def test_capm_formula_matches_hand_calculation(self):
        betas = [1.00, 0.95, 0.85, 0.90, 1.20]
        result = GLOBAL_CAPM.price(capm_inputs(betas))
        expected = [RF + b * ERP for b in betas]
        assert np.allclose(result.expected_returns, expected, atol=1e-12)

    def test_beta_one_zero_alpha_earns_rf_plus_erp(self):
        result = GLOBAL_CAPM.price(capm_inputs([1.0]))
        assert np.isclose(result.expected_returns[0], RF + ERP, atol=1e-12)

    def test_beta_zero_earns_risk_free_rate(self):
        result = GLOBAL_CAPM.price(capm_inputs([0.0]))
        assert np.isclose(result.expected_returns[0], RF, atol=1e-12)

    def test_alpha_is_added_on_top_of_capm(self):
        betas = [1.0, 0.8]
        alphas = [0.01, -0.02]
        result = GLOBAL_CAPM.price(capm_inputs(betas, alphas))
        expected = np.array([RF + ERP + 0.01, RF + 0.8 * ERP - 0.02])
        assert np.allclose(result.expected_returns, expected, atol=1e-12)

    def test_alpha_defaults_to_zero(self):
        without = GLOBAL_CAPM.price(capm_inputs([1.1, 0.9])).expected_returns
        with_zero = GLOBAL_CAPM.price(capm_inputs([1.1, 0.9], [0.0, 0.0])).expected_returns
        assert np.allclose(without, with_zero, atol=1e-12)

    def test_sml_linearity_excess_return_proportional_to_beta(self):
        betas = np.array([0.5, 0.9, 1.3])
        result = GLOBAL_CAPM.price(capm_inputs(betas))
        excess = result.expected_returns - RF
        assert np.allclose(excess / betas, ERP, atol=1e-12)

    def test_decomposition_sums_to_expected_return(self):
        betas = [1.0, 0.85, 1.2]
        alphas = [0.005, 0.0, -0.01]
        result = GLOBAL_CAPM.price(capm_inputs(betas, alphas))
        assert result.factor_contributions is not None
        assert set(result.factor_contributions) == {"erp"}
        reconstructed = (
            result.risk_free_rate
            + sum(result.factor_contributions.values())
            + result.alpha_component
        )
        assert np.allclose(reconstructed, result.expected_returns, atol=1e-12)
        assert np.allclose(result.factor_contributions["erp"], np.array(betas) * ERP)

    def test_default_preset_capm_returns(self):
        preset = get_default_preset()
        result = GLOBAL_CAPM.price(
            PricingInputs(
                risk_free_rate=preset.risk_free_rate,
                factor_premia={"erp": preset.erp},
                loadings=preset.resolved_betas().reshape(-1, 1),
                alphas=preset.resolved_alphas(),
            )
        )
        # US 1.00, Europe ex-UK 0.95, Japan 0.85, UK 0.90, Emerging Markets 1.20 @ 3% Rf, 4.5% ERP
        assert np.allclose(
            result.expected_returns,
            [0.075, 0.07275, 0.06825, 0.0705, 0.084],
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Baked-in betas / default preset wiring
# ---------------------------------------------------------------------------


class TestBakedInDefaults:
    def test_default_preset_betas_match_baked_in_table(self):
        preset = get_default_preset()
        for name, beta in zip(preset.names, preset.resolved_betas()):
            assert np.isclose(beta, DEFAULT_BETA_BY_MARKET[name])

    def test_default_preset_has_all_five_markets_with_betas(self):
        preset = get_default_preset()
        assert set(preset.names) == {"US", "Europe ex-UK", "Japan", "UK", "Emerging Markets"}
        assert preset.betas is not None and preset.betas.shape == (5,)

    def test_default_preset_alphas_are_zero(self):
        preset = get_default_preset()
        assert np.allclose(preset.resolved_alphas(), 0.0)

    def test_default_erp_is_4_5_percent(self):
        assert np.isclose(DEFAULT_ERP, 0.045)
        assert np.isclose(get_default_preset().erp, 0.045)

    def test_betas_are_ordered_sensibly(self):
        # Emerging markets should be the most volatile to the global market; Japan the least.
        assert DEFAULT_BETA_BY_MARKET["Emerging Markets"] > DEFAULT_BETA_BY_MARKET["US"]
        assert DEFAULT_BETA_BY_MARKET["Japan"] < DEFAULT_BETA_BY_MARKET["US"]


# ---------------------------------------------------------------------------
# Pluggable abstraction: multi-factor linear models
# ---------------------------------------------------------------------------

TWO_FACTOR = LinearFactorModel(
    key="test_two_factor",
    label="Test Two-Factor",
    _factors=(
        FactorSpec(key="mkt", label="Market", default_premium=0.05),
        FactorSpec(key="size", label="Size", default_premium=0.02),
    ),
)


class TestMultiFactorLinearModel:
    def test_two_factor_math(self):
        loadings = np.array([[1.0, 0.5], [0.8, -0.3]])
        inputs = PricingInputs(
            risk_free_rate=0.02,
            factor_premia={"mkt": 0.05, "size": 0.02},
            loadings=loadings,
            alphas=np.array([0.0, 0.01]),
        )
        result = TWO_FACTOR.price(inputs)
        expected = np.array(
            [
                0.02 + 1.0 * 0.05 + 0.5 * 0.02,
                0.02 + 0.8 * 0.05 - 0.3 * 0.02 + 0.01,
            ]
        )
        assert np.allclose(result.expected_returns, expected, atol=1e-12)
        assert set(result.factor_contributions) == {"mkt", "size"}

    def test_factor_order_follows_model_spec_not_dict_order(self):
        loadings = np.array([[1.0, 2.0]])
        a = TWO_FACTOR.price(
            PricingInputs(0.0, {"mkt": 0.1, "size": 0.2}, loadings)
        ).expected_returns
        b = TWO_FACTOR.price(
            PricingInputs(0.0, {"size": 0.2, "mkt": 0.1}, loadings)
        ).expected_returns
        assert np.allclose(a, b)

    def test_factors_property_exposes_specs(self):
        assert [f.key for f in TWO_FACTOR.factors] == ["mkt", "size"]
        assert GLOBAL_CAPM.factors[0].key == "erp"


# ---------------------------------------------------------------------------
# Pluggable abstraction: non-linear / non-decomposable models
# ---------------------------------------------------------------------------


class MeanReversionToyModel(AssetPricingModel):
    """Toy non-linear model: E[R_i] = rf + erp * tanh(beta_i). No additive decomposition."""

    key = "test_nonlinear"
    label = "Toy Non-Linear"

    def price(self, inputs: PricingInputs) -> PricingResult:
        betas = np.asarray(inputs.loadings, dtype=float)[:, 0]
        erp = float(inputs.factor_premia["erp"])
        mu = inputs.risk_free_rate + erp * np.tanh(betas)
        return PricingResult(expected_returns=mu, risk_free_rate=inputs.risk_free_rate)


class TestNonLinearExtensibility:
    def test_nonlinear_model_returns_mu_without_decomposition(self):
        model = MeanReversionToyModel()
        result = model.price(capm_inputs([0.5, 2.0]))
        assert result.factor_contributions is None
        assert result.alpha_component is None
        assert np.allclose(
            result.expected_returns,
            RF + ERP * np.tanh(np.array([0.5, 2.0])),
            atol=1e-12,
        )

    def test_expected_returns_convenience_matches_price(self):
        model = MeanReversionToyModel()
        inputs = capm_inputs([0.7, 1.4])
        assert np.allclose(model.expected_returns(inputs), model.price(inputs).expected_returns)

    def test_nonlinear_model_can_be_registered_and_looked_up(self, monkeypatch):
        monkeypatch.setitem(PRICING_MODELS, MeanReversionToyModel.key, MeanReversionToyModel())
        assert get_pricing_model("test_nonlinear").label == "Toy Non-Linear"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_global_capm_is_registered_by_default(self):
        keys = [m.key for m in list_pricing_models()]
        assert "global_capm" in keys
        assert get_pricing_model("global_capm") is GLOBAL_CAPM

    def test_unknown_model_key_raises(self):
        with pytest.raises(ValueError, match="Unknown pricing model"):
            get_pricing_model("does_not_exist")

    def test_register_replaces_same_key(self, monkeypatch):
        monkeypatch.setitem(PRICING_MODELS, TWO_FACTOR.key, TWO_FACTOR)
        replacement = LinearFactorModel(
            key=TWO_FACTOR.key,
            label="Replacement",
            _factors=(FactorSpec(key="mkt", label="Market", default_premium=0.04),),
        )
        register_pricing_model(replacement)
        assert get_pricing_model(TWO_FACTOR.key).label == "Replacement"

    def test_register_rejects_empty_key(self):
        bad = LinearFactorModel(key="", label="No key", _factors=(FactorSpec("x", "X", 0.0),))
        with pytest.raises(ValueError, match="non-empty"):
            register_pricing_model(bad)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_factor_premium_raises(self):
        with pytest.raises(ValueError, match="Missing risk premium"):
            GLOBAL_CAPM.price(PricingInputs(RF, {}, np.array([[1.0]])))

    def test_non_finite_premium_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            GLOBAL_CAPM.price(PricingInputs(RF, {"erp": np.nan}, np.array([[1.0]])))

    def test_non_finite_risk_free_rate_raises(self):
        with pytest.raises(ValueError, match="Risk-free rate must be finite"):
            GLOBAL_CAPM.price(PricingInputs(np.inf, {"erp": ERP}, np.array([[1.0]])))

    def test_missing_loadings_raises(self):
        with pytest.raises(ValueError, match="Factor loadings are required"):
            GLOBAL_CAPM.price(PricingInputs(RF, {"erp": ERP}, None))

    def test_wrong_loading_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            TWO_FACTOR.price(
                PricingInputs(RF, {"mkt": 0.05, "size": 0.02}, np.array([[1.0]]))
            )

    def test_non_finite_loading_raises(self):
        with pytest.raises(ValueError, match="Factor loadings must be finite"):
            GLOBAL_CAPM.price(PricingInputs(RF, {"erp": ERP}, np.array([[np.nan]])))

    def test_alpha_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Alphas must have length"):
            GLOBAL_CAPM.price(capm_inputs([1.0, 0.9], alphas=[0.01]))

    def test_non_finite_alpha_raises(self):
        with pytest.raises(ValueError, match="Alphas must be finite"):
            GLOBAL_CAPM.price(capm_inputs([1.0], alphas=[np.inf]))

    def test_zero_markets_raises(self):
        with pytest.raises(ValueError, match="At least one market"):
            GLOBAL_CAPM.price(PricingInputs(RF, {"erp": ERP}, np.zeros((0, 1))))

    def test_model_without_factors_raises(self):
        empty = LinearFactorModel(key="empty", label="Empty")
        with pytest.raises(ValueError, match="no factors"):
            empty.price(PricingInputs(RF, {}, np.zeros((1, 0))))
