"""Unit tests for the Flet UI controller, pricing tab, and Matplotlib renderers."""

from unittest.mock import MagicMock

import flet as ft
import numpy as np
import pytest

from src.data import DEFAULT_BETA_BY_MARKET, get_default_preset
from src.optimizer import (
    build_covariance_matrix,
    compute_efficient_frontier,
    find_cal_portfolio,
    find_min_variance_portfolio,
    find_tangency_portfolio,
)
from src.pricing import (
    PRICING_MODELS,
    AssetPricingModel,
    FactorSpec,
    LinearFactorModel,
    PricingInputs,
    PricingResult,
)
from src.ui import PortfolioOptimizerView, render_frontier_chart_png, render_pricing_chart_png

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_mock_page() -> MagicMock:
    page = MagicMock(spec=ft.Page)
    page.services = []
    page.overlay = []
    return page


def _make_view() -> PortfolioOptimizerView:
    return PortfolioOptimizerView(_make_mock_page())


# ---------------------------------------------------------------------------
# Chart renderers
# ---------------------------------------------------------------------------


def test_render_frontier_chart_png_produces_valid_png():
    preset = get_default_preset()
    cov = build_covariance_matrix(preset.corr_matrix, volatilities=preset.volatilities)
    mvp = find_min_variance_portfolio(cov)
    tangency = find_tangency_portfolio(preset.expected_returns, cov, preset.risk_free_rate)
    frontier = compute_efficient_frontier(preset.expected_returns, cov, num_points=25)
    cal = find_cal_portfolio(tangency, preset.risk_free_rate, target_return=0.10)

    png_bytes = render_frontier_chart_png(preset, mvp, tangency, frontier, cal)
    assert png_bytes.startswith(PNG_MAGIC)
    assert len(png_bytes) > 5000


def test_render_pricing_chart_sml_for_single_factor():
    from src.pricing import GLOBAL_CAPM

    preset = get_default_preset()
    betas = preset.resolved_betas()
    result = GLOBAL_CAPM.price(
        PricingInputs(preset.risk_free_rate, {"erp": preset.erp}, betas.reshape(-1, 1))
    )
    png = render_pricing_chart_png(
        preset.names, result, betas.reshape(-1, 1), preset.risk_free_rate, "Global CAPM", preset.erp
    )
    assert png is not None and png.startswith(PNG_MAGIC)


def test_render_pricing_chart_stacked_for_multi_factor():
    model = LinearFactorModel(
        key="chart_test",
        label="Chart Test",
        _factors=(FactorSpec("mkt", "Market", 0.05), FactorSpec("size", "Size", 0.02)),
    )
    loadings = np.array([[1.0, 0.5], [0.8, -0.2]])
    result = model.price(PricingInputs(0.02, {"mkt": 0.05, "size": 0.02}, loadings))
    png = render_pricing_chart_png(
        ["A", "B"], result, loadings, 0.02, "Chart Test", 0.05
    )
    assert png is not None and png.startswith(PNG_MAGIC)


def test_render_pricing_chart_returns_none_without_decomposition():
    result = PricingResult(expected_returns=np.array([0.05]), risk_free_rate=0.02)
    assert render_pricing_chart_png(["A"], result, None, 0.02, "Opaque", None) is None


# ---------------------------------------------------------------------------
# Two-tab structure & defaults
# ---------------------------------------------------------------------------


def test_view_has_two_tabs_with_pricing_first():
    view = _make_view()
    segments = view._tab_selector.segments
    assert [s.value for s in segments] == ["pricing", "optimizer"]
    assert view._pricing_tab.visible is True
    assert view._optimizer_tab.visible is False


def test_tab_switch_toggles_visibility():
    view = _make_view()
    event = MagicMock()
    event.control.selected = ["optimizer"]
    view._on_tab_changed(event)
    assert view._pricing_tab.visible is False
    assert view._optimizer_tab.visible is True

    event.control.selected = ["pricing"]
    view._on_tab_changed(event)
    assert view._pricing_tab.visible is True
    assert view._optimizer_tab.visible is False


def test_pricing_tab_defaults_to_global_capm_with_baked_in_betas():
    view = _make_view()
    assert view._pricing_model_key == "global_capm"
    assert view._pricing_model_dropdown.value == "global_capm"
    assert len(view._pricing_result_texts) == 5
    # US: 3% + 1.00 * 4.5% = 7.50%, Emerging Markets: 3% + 1.20 * 4.5% = 8.40%
    assert view._pricing_result_texts[0].value == "7.50"
    assert view._pricing_result_texts[4].value == "8.40"
    betas_shown = [float(f.value) for f in view._pricing_loading_fields["erp"]]
    expected_betas = [DEFAULT_BETA_BY_MARKET[n] for n in get_default_preset().names]
    assert np.allclose(betas_shown, expected_betas)
    assert view._pricing_premia_fields["erp"].value == "4.50"


def test_pricing_chart_is_rendered_on_load():
    view = _make_view()
    assert isinstance(view._pricing_chart_image.src, bytes)
    assert view._pricing_chart_image.src.startswith(PNG_MAGIC)


# ---------------------------------------------------------------------------
# Live-sync feed-forward & Risk-Free Rate sync
# ---------------------------------------------------------------------------


def test_editing_beta_updates_pricing_output_live():
    view = _make_view()
    view._pricing_loading_fields["erp"][0].value = "1.50"
    view._on_pricing_input_changed(None)
    # 3% + 1.50 * 4.5% = 9.75%
    assert view._pricing_result_texts[0].value == "9.75"


def test_live_sync_off_leaves_optimizer_returns_editable_and_unchanged():
    view = _make_view()
    before = [tf.value for tf in view._return_fields]
    view._pricing_loading_fields["erp"][0].value = "1.50"
    view._on_pricing_input_changed(None)
    assert [tf.value for tf in view._return_fields] == before
    assert all(tf.read_only is False for tf in view._return_fields)


def test_live_sync_on_mirrors_pricing_returns_into_optimizer_and_locks_fields():
    view = _make_view()
    view._live_sync_cb.value = True
    view._on_live_sync_changed(None)

    assert all(tf.read_only is True for tf in view._return_fields)
    for tf, text in zip(view._return_fields, [t.value for t in view._pricing_result_texts]):
        assert tf.value == text

    # Changing a beta in Tab 1 flows through to Tab 2 returns
    view._pricing_loading_fields["erp"][1].value = "0.50"
    view._on_pricing_input_changed(None)
    assert view._return_fields[1].value == f"{(0.03 + 0.5 * 0.045) * 100:.2f}"


def test_live_sync_turned_off_unlocks_optimizer_returns():
    view = _make_view()
    view._live_sync_cb.value = True
    view._on_live_sync_changed(None)
    view._live_sync_cb.value = False
    view._on_live_sync_changed(None)
    assert all(tf.read_only is False for tf in view._return_fields)


def test_live_sync_feeds_optimizer_run_with_pricing_returns():
    view = _make_view()
    view._live_sync_cb.value = True
    view._on_live_sync_changed(None)
    view.run_optimization()
    assert np.allclose(
        view._last_tangency.weights.sum(), 1.0, atol=1e-6
    )
    # Optimizer must have consumed the CAPM-derived returns, not the Phase 1 placeholders
    dataset, _ = view._parse_inputs()
    # Optimizer tab displays returns to 2 decimals (in %), so allow 0.005% rounding.
    assert np.allclose(
        dataset.expected_returns,
        [0.075, 0.07275, 0.06825, 0.0705, 0.084],
        atol=5e-5,
    )


def test_risk_free_rate_syncs_from_pricing_tab_to_optimizer():
    view = _make_view()
    view._pricing_rf_field.value = "4.25"
    view._on_pricing_rf_changed(None)
    assert view._rf_field.value == "4.25"


def test_risk_free_rate_syncs_from_optimizer_tab_to_pricing():
    view = _make_view()
    view._rf_field.value = "2.10"
    view._on_optimizer_rf_changed(None)
    assert view._pricing_rf_field.value == "2.10"
    assert np.isclose(view._pricing_last_result.risk_free_rate, 0.021)


def test_rf_change_flows_to_pricing_outputs():
    view = _make_view()
    view._pricing_rf_field.value = "5.00"
    view._on_pricing_rf_changed(None)
    # US: 5% + 1.00 * 4.5% = 9.50%
    assert view._pricing_result_texts[0].value == "9.50"


# ---------------------------------------------------------------------------
# Input validation on the pricing tab
# ---------------------------------------------------------------------------


def test_invalid_beta_shows_inline_error_and_blanks_outputs():
    view = _make_view()
    view._pricing_loading_fields["erp"][0].value = "abc"
    view._on_pricing_input_changed(None)
    assert "Loading" in view._pricing_status.value or "valid number" in view._pricing_status.value
    assert all(t.value == "—" for t in view._pricing_result_texts)
    assert view._pricing_last_result is None


def test_invalid_pricing_input_does_not_crash_optimizer_when_sync_off():
    view = _make_view()
    view._pricing_loading_fields["erp"][0].value = "abc"
    view._on_pricing_input_changed(None)
    # Optimizer runs off its own (Tab 2) inputs and should still succeed
    view.run_optimization()
    assert view._last_tangency is not None


def test_save_path_reads_betas_and_alphas_from_pricing_tab():
    view = _make_view()
    view._pricing_loading_fields["erp"][2].value = "0.70"
    view._pricing_alpha_fields[2].value = "0.50"  # percent
    dataset, _ = view._parse_inputs(include_pricing=True)
    assert np.isclose(dataset.resolved_betas()[2], 0.70)
    assert np.isclose(dataset.resolved_alphas()[2], 0.005)
    assert np.isclose(dataset.erp, 0.045)


# ---------------------------------------------------------------------------
# Model switching: extensibility through the registry
# ---------------------------------------------------------------------------


class _OpaqueNonLinearModel(AssetPricingModel):
    """Non-additive model: no decomposition, no factor loadings in the UI."""

    key = "test_opaque"
    label = "Opaque Non-Linear"

    def price(self, inputs: PricingInputs) -> PricingResult:
        mu = np.full(len(inputs.alphas), inputs.risk_free_rate + 0.05)
        return PricingResult(expected_returns=mu, risk_free_rate=inputs.risk_free_rate)


def test_switching_to_nonlinear_model_renders_without_decomposition(monkeypatch):
    monkeypatch.setitem(PRICING_MODELS, _OpaqueNonLinearModel.key, _OpaqueNonLinearModel())
    view = _make_view()
    view._pricing_model_dropdown.value = "test_opaque"
    view._on_pricing_model_changed(None)

    assert view._pricing_loading_fields == {}
    assert all(t.value == "8.00" for t in view._pricing_result_texts)  # 3% + 5%
    assert "no chart" in view._pricing_chart_note.value.lower()
    assert view._pricing_chart_image.src == b""


def test_switching_models_preserves_per_model_inputs(monkeypatch):
    monkeypatch.setitem(PRICING_MODELS, _OpaqueNonLinearModel.key, _OpaqueNonLinearModel())
    view = _make_view()

    # Edit CAPM beta, switch away, switch back: edit should survive
    view._pricing_loading_fields["erp"][0].value = "1.60"
    view._on_pricing_input_changed(None)
    view._pricing_model_dropdown.value = "test_opaque"
    view._on_pricing_model_changed(None)
    view._pricing_model_dropdown.value = "global_capm"
    view._on_pricing_model_changed(None)

    assert view._pricing_loading_fields["erp"][0].value == "1.60"
    # 3% + 1.60 * 4.5% = 10.20%
    assert view._pricing_result_texts[0].value == "10.20"


def test_multi_factor_model_builds_one_loading_column_per_factor(monkeypatch):
    multi = LinearFactorModel(
        key="test_multi",
        label="Test Multi",
        _factors=(FactorSpec("mkt", "Market", 0.05), FactorSpec("size", "Size", 0.02)),
    )
    monkeypatch.setitem(PRICING_MODELS, multi.key, multi)
    view = _make_view()
    view._pricing_model_dropdown.value = "test_multi"
    view._on_pricing_model_changed(None)

    assert set(view._pricing_loading_fields) == {"mkt", "size"}
    assert set(view._pricing_premia_fields) == {"mkt", "size"}
    assert len(view._pricing_loading_fields["mkt"]) == 5
    assert view._pricing_chart_image.src.startswith(PNG_MAGIC)
    assert view._pricing_status.value == ""


def test_model_switch_ignores_premia_from_other_models(monkeypatch):
    multi = LinearFactorModel(
        key="test_multi",
        label="Test Multi",
        _factors=(FactorSpec("mkt", "Market", 0.05), FactorSpec("size", "Size", 0.02)),
    )
    monkeypatch.setitem(PRICING_MODELS, multi.key, multi)
    view = _make_view()
    view._pricing_model_dropdown.value = "test_multi"
    view._on_pricing_model_changed(None)
    # Defaults for the new model come from its FactorSpecs
    assert view._pricing_premia_fields["mkt"].value == "5.00"
    assert view._pricing_premia_fields["size"].value == "2.00"


# ---------------------------------------------------------------------------
# Phase 1/2 behaviour preserved on the optimizer tab
# ---------------------------------------------------------------------------


def test_flet_ui_correlation_mirroring_and_cal_modes():
    view = _make_view()

    assert isinstance(view._chart_image.src, bytes)
    assert view._chart_image.src.startswith(PNG_MAGIC)
    assert view._last_tangency is not None
    assert view._last_cal is None

    view._corr_fields[0][1].value = "0.42"
    view._mirror_corr_cell(0, 1)
    assert view._corr_fields[1][0].value == "0.42"

    assert view._corr_fields[0][0].read_only is True
    assert view._corr_fields[0][0].value == "1.00"

    view._cal_mode_group.value = "return"
    view._on_cal_mode_changed(None)
    assert view._cal_target_field.disabled is False
    view._cal_target_field.value = "10.0"
    view.run_optimization()
    assert view._last_cal is not None
    assert np.isclose(view._last_cal.expected_return, 0.10)
    assert len(view._weights_table.rows) == 6  # 5 markets + Risk-Free Asset

    view._cal_mode_group.value = "volatility"
    view._on_cal_mode_changed(None)
    view._cal_target_field.value = "12.0"
    view.run_optimization()
    assert view._last_cal is not None
    assert np.isclose(view._last_cal.volatility, 0.12)


def test_flet_ui_edge_cases_show_dialog_without_crashing():
    page = _make_mock_page()
    view = PortfolioOptimizerView(page)
    page.show_dialog.reset_mock()

    # 1. Non-numeric optimizer input
    view._return_fields[0].value = "abc"
    view.run_optimization()
    assert page.show_dialog.call_count == 1
    page.show_dialog.reset_mock()

    # 2. Out-of-bounds correlation
    view.load_dataset(get_default_preset())
    view._corr_fields[0][1].value = "1.5"
    view._mirror_corr_cell(0, 1)
    view.run_optimization()
    assert page.show_dialog.call_count == 1
    page.show_dialog.reset_mock()

    # 3. Non-PSD correlation matrix
    view.load_dataset(get_default_preset())
    view._corr_fields[0][1].value = "0.95"
    view._mirror_corr_cell(0, 1)
    view._corr_fields[1][2].value = "0.95"
    view._mirror_corr_cell(1, 2)
    view._corr_fields[0][2].value = "-0.95"
    view._mirror_corr_cell(0, 2)
    view.run_optimization()
    assert page.show_dialog.call_count == 1
    page.show_dialog.reset_mock()

    # 4. Risk-free rate higher than all expected returns
    view.load_dataset(get_default_preset())
    view._rf_field.value = "15.0"
    view.run_optimization()
    assert page.show_dialog.call_count == 1
    dlg = page.show_dialog.call_args[0][0]
    assert isinstance(dlg, ft.AlertDialog)
    assert "Risk-Free Rate Notice" in dlg.title.value


def test_market_name_edit_updates_pricing_labels():
    view = _make_view()
    view._name_fields[0].value = "United States"
    view._on_market_name_edit(0)
    assert view._pricing_name_texts[0].value == "United States"
    assert view._corr_row_labels[0].value == view._short_label("United States")
