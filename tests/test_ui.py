"""Unit tests for the Flet UI controller and Matplotlib PNG renderer."""

from unittest.mock import MagicMock

import flet as ft
import numpy as np

from src.data import get_default_preset
from src.optimizer import (
    build_covariance_matrix,
    compute_efficient_frontier,
    find_cal_portfolio,
    find_min_variance_portfolio,
    find_tangency_portfolio,
)
from src.ui import PortfolioOptimizerView, render_frontier_chart_png


def _make_mock_page() -> MagicMock:
    page = MagicMock(spec=ft.Page)
    page.services = []
    page.overlay = []
    return page


def test_render_frontier_chart_png_produces_valid_png():
    preset = get_default_preset()
    cov = build_covariance_matrix(preset.corr_matrix, volatilities=preset.volatilities)
    mvp = find_min_variance_portfolio(cov)
    tangency = find_tangency_portfolio(preset.expected_returns, cov, preset.risk_free_rate)
    frontier = compute_efficient_frontier(preset.expected_returns, cov, num_points=25)
    cal = find_cal_portfolio(tangency, preset.risk_free_rate, target_return=0.10)

    png_bytes = render_frontier_chart_png(preset, mvp, tangency, frontier, cal)
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 5000


def test_flet_ui_correlation_mirroring_and_cal_modes():
    page = _make_mock_page()
    view = PortfolioOptimizerView(page)

    # Initial optimization ran and produced PNG bytes
    assert isinstance(view._chart_image.src, bytes)
    assert view._chart_image.src.startswith(b"\x89PNG\r\n\x1a\n")
    assert view._last_tangency is not None
    assert view._last_cal is None

    # Auto-mirroring upper -> lower triangle
    view._corr_fields[0][1].value = "0.42"
    view._mirror_corr_cell(0, 1)
    assert view._corr_fields[1][0].value == "0.42"

    # Diagonal is read-only 1.00
    assert view._corr_fields[0][0].read_only is True
    assert view._corr_fields[0][0].value == "1.00"

    # CAL target return mode
    view._cal_mode_group.value = "return"
    view._on_cal_mode_changed(None)
    assert view._cal_target_field.disabled is False
    view._cal_target_field.value = "10.0"
    view.run_optimization()
    assert view._last_cal is not None
    assert np.isclose(view._last_cal.expected_return, 0.10)
    assert len(view._weights_table.rows) == 6  # 5 markets + Risk-Free Asset

    # CAL target volatility mode
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

    # 1. Non-numeric input
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
