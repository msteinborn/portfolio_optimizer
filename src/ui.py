"""Flet v1.0+ UI and Matplotlib visualization for the Portfolio Optimizer.

Two views (switched via a segmented control):
  1. Asset Pricing: pluggable pricing model -> expected returns (mu).
  2. Portfolio Optimizer: MVO, Efficient Frontier, CAL target portfolio.

Designed to run both locally and as a static Pyodide web app via `flet publish`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import flet as ft
import matplotlib
import numpy as np

matplotlib.use("agg")
from matplotlib.figure import Figure

from src.data import (
    DEFAULT_ERP,
    MarketDataset,
    get_default_preset,
    parse_market_assumptions_csv_text,
    serialize_market_assumptions_csv,
    serialize_weights_csv,
)
from src.optimizer import (
    CALPortfolioResult,
    EfficientFrontierResult,
    PortfolioResult,
    build_covariance_matrix,
    compute_efficient_frontier,
    find_cal_portfolio,
    find_min_variance_portfolio,
    find_tangency_portfolio,
    portfolio_return,
)
from src.pricing import (
    GLOBAL_CAPM,
    AssetPricingModel,
    LinearFactorModel,
    PricingInputs,
    PricingResult,
    get_pricing_model,
    list_pricing_models,
)

GLOBAL_CAPM_KEY = GLOBAL_CAPM.key
CAPM_FACTOR_KEY = GLOBAL_CAPM.factors[0].key

_CONTRIB_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def render_frontier_chart_png(
    dataset: MarketDataset,
    mvp: PortfolioResult,
    tangency: PortfolioResult,
    frontier: EfficientFrontierResult,
    cal_result: Optional[CALPortfolioResult] = None,
) -> bytes:
    """Render the Efficient Frontier & CAL Matplotlib figure to PNG bytes."""
    fig = Figure(figsize=(7.6, 4.5), dpi=130)
    ax = fig.add_subplot(111)

    ax.plot(
        frontier.volatilities * 100.0,
        frontier.returns * 100.0,
        color="#1f77b4",
        linewidth=2.2,
        label="Efficient Frontier",
    )

    vols_pct = dataset.volatilities * 100.0
    rets_pct = dataset.expected_returns * 100.0
    ax.scatter(vols_pct, rets_pct, color="#555555", s=45, zorder=4, label="Individual Markets")
    for name, x, y in zip(dataset.names, vols_pct, rets_pct):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.5)

    mvp_ret = portfolio_return(mvp.weights, dataset.expected_returns)
    ax.scatter(
        [mvp.volatility * 100.0],
        [mvp_ret * 100.0],
        color="#2ca02c",
        marker="D",
        s=80,
        zorder=5,
        label="Min Variance Portfolio (MVP)",
    )

    tang_ret = tangency.expected_return if tangency.expected_return is not None else 0.0
    ax.scatter(
        [tangency.volatility * 100.0],
        [tang_ret * 100.0],
        color="#d62728",
        marker="*",
        s=220,
        zorder=6,
        label="Tangency Portfolio (Max Sharpe)",
    )

    cal_vol_max = cal_result.volatility if cal_result is not None else 0.0
    max_x = (
        max(
            float(np.max(frontier.volatilities)),
            float(np.max(dataset.volatilities)),
            tangency.volatility,
            cal_vol_max,
        )
        * 1.15
    )
    if tangency.volatility > 1e-12:
        slope = (tang_ret - dataset.risk_free_rate) / tangency.volatility
        cal_x = np.array([0.0, max_x])
        cal_y = dataset.risk_free_rate + slope * cal_x
        ax.plot(
            cal_x * 100.0,
            cal_y * 100.0,
            linestyle="--",
            color="#ff7f0e",
            linewidth=1.6,
            label="Capital Allocation Line (CAL)",
        )

    if cal_result is not None:
        ax.scatter(
            [cal_result.volatility * 100.0],
            [cal_result.expected_return * 100.0],
            color="#9467bd",
            marker="o",
            s=130,
            edgecolors="black",
            linewidths=1.0,
            zorder=7,
            label="CAL Target Portfolio",
        )

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Expected Return (%)")
    ax.set_title("Efficient Frontier & Capital Allocation Line")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    fig.clf()
    return buf.getvalue()


def render_pricing_chart_png(
    names: list[str],
    result: PricingResult,
    loadings: Optional[np.ndarray],
    risk_free_rate: float,
    model_label: str,
    primary_premium: Optional[float],
) -> Optional[bytes]:
    """Render the pricing chart for a model's output.

    - Single-factor models with a decomposition: Security Market Line (beta vs. E[R]).
    - Multi-factor models with a decomposition: stacked bars of factor contributions.
    - Models without a decomposition: returns None (no chart).
    """
    if not result.factor_contributions:
        return None

    fig = Figure(figsize=(7.6, 4.5), dpi=130)
    ax = fig.add_subplot(111)
    expected_pct = result.expected_returns * 100.0

    if len(result.factor_contributions) == 1 and loadings is not None and primary_premium is not None:
        betas = np.asarray(loadings, dtype=float)[:, 0]
        x_max = max(float(np.max(np.abs(betas))), 1e-6) * 1.2
        xs = np.array([0.0, x_max])
        ax.plot(
            xs,
            (risk_free_rate + primary_premium * xs) * 100.0,
            linestyle="--",
            color="#ff7f0e",
            linewidth=1.8,
            label="Security Market Line",
        )
        ax.scatter(betas, expected_pct, color="#1f77b4", s=60, zorder=4, label="Markets (E[R])")
        for name, x, y in zip(names, betas, expected_pct):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
        ax.scatter([0.0], [risk_free_rate * 100.0], color="#555555", marker="s", s=60, zorder=5, label="Risk-free rate")
        ax.set_xlabel("Beta")
        ax.set_ylabel("Expected Return (%)")
        ax.set_title(f"Security Market Line — {model_label}")
    else:
        x = np.arange(len(names))
        bottom = np.zeros(len(names))
        for idx, (key, contrib) in enumerate(result.factor_contributions.items()):
            vals = np.asarray(contrib, dtype=float) * 100.0
            ax.bar(x, vals, bottom=bottom, label=key, color=_CONTRIB_COLORS[idx % len(_CONTRIB_COLORS)])
            bottom = bottom + vals
        if result.alpha_component is not None:
            alpha_vals = np.asarray(result.alpha_component, dtype=float) * 100.0
            ax.bar(x, alpha_vals, bottom=bottom, label="Alpha", color="#999999")
        ax.axhline(risk_free_rate * 100.0, linestyle="--", color="#333333", label="Risk-free rate")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Return contribution (%)")
        ax.set_title(f"Factor Return Decomposition — {model_label}")

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    fig.clf()
    return buf.getvalue()


class PortfolioOptimizerView:
    """Flet UI controller: Asset Pricing tab + Portfolio Optimizer tab."""

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._ready = False
        self._syncing_corr = False
        self._syncing_rf = False

        # --- Portfolio Optimizer tab state ---
        self._name_fields: list[ft.TextField] = []
        self._return_fields: list[ft.TextField] = []
        self._risk_fields: list[ft.TextField] = []
        self._corr_fields: list[list[ft.TextField]] = []
        self._corr_col_labels: list[ft.Text] = []
        self._corr_row_labels: list[ft.Text] = []
        self._last_names: list[str] = []
        self._last_tangency: Optional[PortfolioResult] = None
        self._last_cal: Optional[CALPortfolioResult] = None

        # --- Asset Pricing tab state ---
        self._pricing_model_key: str = GLOBAL_CAPM_KEY
        # model key -> raw text values, so per-model inputs survive model switches
        self._pricing_state: dict[str, dict] = {}
        self._pricing_built = False
        self._pricing_premia_fields: dict[str, ft.TextField] = {}
        self._pricing_loading_fields: dict[str, list[ft.TextField]] = {}
        self._pricing_alpha_fields: list[ft.TextField] = []
        self._pricing_result_texts: list[ft.Text] = []
        self._pricing_name_texts: list[ft.Text] = []
        self._pricing_last_result: Optional[PricingResult] = None

        self.file_picker = ft.FilePicker()
        self.page.services.append(self.file_picker)

        self._configure_page()
        self._build_layout()
        self.load_dataset(get_default_preset())
        self.run_optimization(update_page=False)
        self._ready = True
        self.page.update()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _configure_page(self) -> None:
        self.page.title = "International Equity Portfolio Optimizer"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.padding = 16
        self.page.scroll = ft.ScrollMode.AUTO

    def _section_card(self, title: str, content: ft.Control) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(title, size=15, weight=ft.FontWeight.BOLD),
                    ft.Divider(height=1),
                    content,
                ],
                spacing=8,
                tight=True,
            ),
            padding=12,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=8,
            bgcolor=ft.Colors.SURFACE,
        )

    def _safe_update(self) -> None:
        if self._ready:
            self.page.update()

    def _build_layout(self) -> None:
        self._tab_selector = ft.SegmentedButton(
            selected=["pricing"],
            on_change=self._on_tab_changed,
            segments=[
                ft.Segment(value="pricing", label=ft.Text("1. Asset Pricing")),
                ft.Segment(value="optimizer", label=ft.Text("2. Portfolio Optimizer")),
            ],
        )

        self._pricing_tab = self._build_pricing_tab()
        self._optimizer_tab = self._build_optimizer_tab()
        self._optimizer_tab.visible = False

        header = ft.Text(
            "International Equity Portfolio Optimizer",
            size=22,
            weight=ft.FontWeight.BOLD,
        )
        self.page.add(header, self._tab_selector, self._pricing_tab, self._optimizer_tab)

    def _on_tab_changed(self, e: ft.ControlEvent) -> None:
        selected = next(iter(e.control.selected), "pricing")
        self._pricing_tab.visible = selected == "pricing"
        self._optimizer_tab.visible = selected == "optimizer"
        self._safe_update()

    def _build_pricing_tab(self) -> ft.ResponsiveRow:
        self._pricing_model_dropdown = ft.Dropdown(
            label="Pricing Model",
            value=self._pricing_model_key,
            options=[ft.DropdownOption(key=m.key, text=m.label) for m in list_pricing_models()],
            width=320,
            on_select=self._on_pricing_model_changed,
        )
        self._pricing_model_desc = ft.Text(size=12, italic=True)
        self._pricing_rf_field = ft.TextField(
            label="Risk-Free Rate (%)",
            value="3.00",
            width=200,
            dense=True,
            on_change=self._on_pricing_rf_changed,
        )
        self._pricing_premia_row = ft.Row(wrap=True, spacing=12)
        self._pricing_table_col = ft.Column(spacing=6, tight=True)
        self._pricing_status = ft.Text(color=ft.Colors.RED, size=12)

        left = ft.Column(
            col={"xs": 12, "lg": 6},
            controls=[
                self._section_card(
                    "Pricing Model",
                    ft.Column(
                        controls=[self._pricing_model_dropdown, self._pricing_model_desc],
                        spacing=6,
                        tight=True,
                    ),
                ),
                self._section_card(
                    "Shared Inputs (base currency)",
                    ft.Column(
                        controls=[
                            self._pricing_rf_field,
                            ft.Text("Factor risk premia", weight=ft.FontWeight.W_600, size=13),
                            self._pricing_premia_row,
                        ],
                        spacing=8,
                        tight=True,
                    ),
                ),
                self._section_card(
                    "Market Inputs & Expected Returns",
                    ft.Column(
                        controls=[
                            ft.Row(
                                controls=[ft.Container(content=self._pricing_table_col)],
                                scroll=ft.ScrollMode.AUTO,
                            ),
                            self._pricing_status,
                        ],
                        spacing=8,
                        tight=True,
                    ),
                ),
            ],
            spacing=12,
        )

        self._pricing_chart_image = ft.Image(src=b"", fit=ft.BoxFit.CONTAIN, height=420)
        self._pricing_chart_note = ft.Text("", size=12, italic=True)
        right = ft.Column(
            col={"xs": 12, "lg": 6},
            controls=[
                self._section_card(
                    "Pricing Chart",
                    ft.Column(
                        controls=[
                            ft.Container(
                                content=self._pricing_chart_image,
                                alignment=ft.Alignment.CENTER,
                            ),
                            self._pricing_chart_note,
                        ],
                        spacing=6,
                        tight=True,
                    ),
                ),
            ],
            spacing=12,
        )

        return ft.ResponsiveRow(controls=[left, right], spacing=14, run_spacing=14)

    def _build_optimizer_tab(self) -> ft.ResponsiveRow:
        self._risk_mode_group = ft.RadioGroup(
            value="volatility",
            on_change=self._on_risk_mode_changed,
            content=ft.Row(
                controls=[
                    ft.Radio(value="volatility", label="Volatility (%)"),
                    ft.Radio(value="variance", label="Variance (σ²)"),
                ],
                spacing=12,
            ),
        )
        self._risk_col_header = ft.Text(
            "Volatility (%)", weight=ft.FontWeight.BOLD, size=12, width=100
        )
        self._markets_rows_col = ft.Column(spacing=6, tight=True)
        markets_content = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("Risk Input:", weight=ft.FontWeight.W_500, size=13),
                        self._risk_mode_group,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self._markets_rows_col,
            ],
            spacing=8,
            tight=True,
        )

        self._corr_grid_col = ft.Column(spacing=4, tight=True)

        self._rf_field = ft.TextField(
            label="Risk-Free Rate (%)",
            value="3.00",
            width=160,
            dense=True,
            on_change=self._on_optimizer_rf_changed,
        )
        self._allow_short_cb = ft.Checkbox(label="Allow Short Selling", value=False)
        self._live_sync_cb = ft.Checkbox(
            label="Use Pricing Model Returns (live-sync from Asset Pricing tab)",
            value=False,
            on_change=self._on_live_sync_changed,
        )

        self._cal_mode_group = ft.RadioGroup(
            value="none",
            on_change=self._on_cal_mode_changed,
            content=ft.Row(
                controls=[
                    ft.Radio(value="none", label="None"),
                    ft.Radio(value="return", label="Target Return (%)"),
                    ft.Radio(value="volatility", label="Target Vol (%)"),
                ],
                wrap=True,
                spacing=8,
            ),
        )
        self._cal_target_field = ft.TextField(
            label="Target Value (%)",
            value="8.00",
            width=180,
            dense=True,
            disabled=True,
        )

        params_content = ft.Column(
            controls=[
                ft.Row(
                    controls=[self._rf_field, self._allow_short_cb],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    wrap=True,
                    spacing=12,
                ),
                self._live_sync_cb,
                ft.Text(
                    "CAL Target Portfolio (Optional)",
                    size=13,
                    weight=ft.FontWeight.W_600,
                ),
                self._cal_mode_group,
                self._cal_target_field,
                ft.Row(
                    controls=[
                        ft.FilledButton(
                            "Run Optimization",
                            icon=ft.Icons.PLAY_ARROW,
                            on_click=lambda _: self.run_optimization(),
                        ),
                        ft.OutlinedButton(
                            "Load Preset",
                            icon=ft.Icons.RESTORE,
                            on_click=lambda _: self._on_load_preset(),
                        ),
                    ],
                    wrap=True,
                    spacing=8,
                ),
                ft.Row(
                    controls=[
                        ft.OutlinedButton(
                            "Import CSV",
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=self._on_import_csv,
                        ),
                        ft.OutlinedButton(
                            "Save CSV",
                            icon=ft.Icons.SAVE,
                            on_click=self._on_save_csv,
                        ),
                    ],
                    wrap=True,
                    spacing=8,
                ),
            ],
            spacing=8,
            tight=True,
        )

        left_col = ft.Column(
            col={"xs": 12, "lg": 5},
            controls=[
                self._section_card("Market Assumptions", markets_content),
                self._section_card(
                    "Correlation Matrix (Auto-Mirrored)", self._corr_grid_col
                ),
                self._section_card("Optimization Parameters", params_content),
            ],
            spacing=12,
        )

        self._chart_image = ft.Image(src=b"", fit=ft.BoxFit.CONTAIN, height=420)
        self._metrics_text = ft.Text(
            "Expected Return: —  |  Volatility: —  |  Sharpe Ratio: —",
            size=13,
            weight=ft.FontWeight.BOLD,
        )
        self._weights_table = ft.DataTable(
            columns=[
                ft.DataColumn(label=ft.Text("Market", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(
                    label=ft.Text("Optimal Weight (%)", weight=ft.FontWeight.BOLD),
                    numeric=True,
                ),
            ],
            rows=[],
            column_spacing=32,
            heading_row_height=36,
            data_row_min_height=34,
        )

        summary_content = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Container(content=self._metrics_text, expand=True),
                        ft.OutlinedButton(
                            "Export Weights to CSV",
                            icon=ft.Icons.DOWNLOAD,
                            on_click=self._on_export_weights_csv,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    controls=[self._weights_table],
                    scroll=ft.ScrollMode.AUTO,
                ),
            ],
            spacing=8,
            tight=True,
        )

        right_col = ft.Column(
            col={"xs": 12, "lg": 7},
            controls=[
                self._section_card(
                    "Mean-Variance Efficient Frontier",
                    ft.Container(
                        content=self._chart_image,
                        alignment=ft.Alignment.CENTER,
                    ),
                ),
                self._section_card("Optimal Portfolio Summary", summary_content),
            ],
            spacing=12,
        )

        return ft.ResponsiveRow(
            controls=[left_col, right_col],
            spacing=14,
            run_spacing=14,
        )

    @staticmethod
    def _short_label(name: str, max_len: int = 8) -> str:
        clean = name.strip()
        return clean if len(clean) <= max_len else clean[: max_len - 1] + "…"

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self, dataset: MarketDataset) -> None:
        """Populate both tabs from a MarketDataset."""
        rf_text = f"{dataset.risk_free_rate * 100.0:.2f}"
        self._syncing_rf = True
        try:
            self._rf_field.value = rf_text
            self._pricing_rf_field.value = rf_text
        finally:
            self._syncing_rf = False

        self._build_markets_table(dataset)
        self._build_corr_grid(dataset)

        betas = dataset.resolved_betas()
        alphas = dataset.resolved_alphas()
        self._pricing_state = {
            GLOBAL_CAPM_KEY: {
                "premia": {CAPM_FACTOR_KEY: f"{dataset.erp * 100.0:.2f}"},
                "loadings": {CAPM_FACTOR_KEY: [f"{b:.2f}" for b in betas]},
                "alphas": [f"{a * 100.0:.2f}" for a in alphas],
            }
        }
        self._rebuild_pricing_panel()

    def _build_markets_table(self, dataset: MarketDataset) -> None:
        self._markets_rows_col.controls.clear()
        self._name_fields = []
        self._return_fields = []
        self._risk_fields = []

        mode = self._risk_mode_group.value or "volatility"
        self._risk_col_header.value = (
            "Volatility (%)" if mode == "volatility" else "Variance (σ²)"
        )

        self._markets_rows_col.controls.append(
            ft.Row(
                controls=[
                    ft.Text("Market Name", weight=ft.FontWeight.BOLD, size=12, width=168),
                    ft.Text("Exp. Ret (%)", weight=ft.FontWeight.BOLD, size=12, width=95),
                    self._risk_col_header,
                ],
                spacing=8,
            )
        )

        live = bool(self._live_sync_cb.value) if hasattr(self, "_live_sync_cb") else False
        for i, name in enumerate(dataset.names):
            name_tf = ft.TextField(
                value=name,
                width=168,
                dense=True,
                content_padding=8,
                on_change=lambda e, idx=i: self._on_market_name_edit(idx),
            )
            ret_tf = ft.TextField(
                value=f"{dataset.expected_returns[i] * 100.0:.2f}",
                width=95,
                dense=True,
                content_padding=8,
                read_only=live,
            )
            risk_str = (
                f"{dataset.volatilities[i] * 100.0:.2f}"
                if mode == "volatility"
                else f"{dataset.variances[i]:.6f}"
            )
            risk_tf = ft.TextField(
                value=risk_str,
                width=100,
                dense=True,
                content_padding=8,
            )
            self._name_fields.append(name_tf)
            self._return_fields.append(ret_tf)
            self._risk_fields.append(risk_tf)
            self._markets_rows_col.controls.append(
                ft.Row(controls=[name_tf, ret_tf, risk_tf], spacing=8)
            )

    def _build_corr_grid(self, dataset: MarketDataset) -> None:
        self._corr_grid_col.controls.clear()
        n = len(dataset.names)
        self._corr_fields = []
        self._corr_col_labels = []
        self._corr_row_labels = []

        header_controls: list[ft.Control] = [ft.Container(width=68)]
        for name in dataset.names:
            lbl = ft.Text(
                self._short_label(name),
                size=11,
                weight=ft.FontWeight.BOLD,
                width=62,
                text_align=ft.TextAlign.CENTER,
            )
            self._corr_col_labels.append(lbl)
            header_controls.append(lbl)
        self._corr_grid_col.controls.append(
            ft.Row(controls=header_controls, spacing=4, scroll=ft.ScrollMode.AUTO)
        )

        for i, name in enumerate(dataset.names):
            row_lbl = ft.Text(
                self._short_label(name),
                size=11,
                weight=ft.FontWeight.BOLD,
                width=68,
                text_align=ft.TextAlign.RIGHT,
            )
            self._corr_row_labels.append(row_lbl)

            row_controls: list[ft.Control] = [row_lbl]
            row_fields: list[ft.TextField] = []
            for j in range(n):
                val = 1.0 if i == j else float(dataset.corr_matrix[i, j])
                tf = ft.TextField(
                    value=f"{val:.2f}",
                    width=62,
                    dense=True,
                    content_padding=6,
                    text_align=ft.TextAlign.CENTER,
                    read_only=(i == j),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST if i == j else None,
                    on_change=(
                        None
                        if i == j
                        else (lambda e, r=i, c=j: self._mirror_corr_cell(r, c))
                    ),
                )
                row_fields.append(tf)
                row_controls.append(tf)
            self._corr_fields.append(row_fields)
            self._corr_grid_col.controls.append(
                ft.Row(controls=row_controls, spacing=4, scroll=ft.ScrollMode.AUTO)
            )

    def _on_market_name_edit(self, idx: int) -> None:
        if idx < len(self._corr_col_labels) and idx < len(self._corr_row_labels):
            name = self._name_fields[idx].value or ""
            short = self._short_label(name)
            self._corr_col_labels[idx].value = short
            self._corr_row_labels[idx].value = short
        if idx < len(self._pricing_name_texts):
            self._pricing_name_texts[idx].value = (self._name_fields[idx].value or "").strip()
        self._safe_update()

    def _mirror_corr_cell(self, row: int, col: int) -> None:
        """Auto-mirror edits between upper and lower triangle of the correlation grid."""
        if self._syncing_corr or row == col:
            return
        self._syncing_corr = True
        try:
            val = self._corr_fields[row][col].value
            if self._corr_fields[col][row].value != val:
                self._corr_fields[col][row].value = val
                self._safe_update()
        finally:
            self._syncing_corr = False

    def _on_risk_mode_changed(self, _e: ft.ControlEvent) -> None:
        mode = self._risk_mode_group.value or "volatility"
        self._risk_col_header.value = (
            "Volatility (%)" if mode == "volatility" else "Variance (σ²)"
        )
        for tf in self._risk_fields:
            try:
                val = float(tf.value or "")
                if mode == "variance":
                    tf.value = f"{(val / 100.0) ** 2:.6f}"
                elif val >= 0.0:
                    tf.value = f"{np.sqrt(val) * 100.0:.2f}"
            except ValueError:
                continue
        self._safe_update()

    def _on_cal_mode_changed(self, _e: ft.ControlEvent) -> None:
        mode = self._cal_mode_group.value or "none"
        if mode == "none":
            self._cal_target_field.disabled = True
            self._cal_target_field.label = "Target Value (%)"
        elif mode == "return":
            self._cal_target_field.disabled = False
            self._cal_target_field.label = "Target Return (%)"
        else:
            self._cal_target_field.disabled = False
            self._cal_target_field.label = "Target Volatility (%)"
        self._safe_update()

    # ------------------------------------------------------------------
    # Asset Pricing tab
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_float(text: Optional[str], label: str) -> float:
        try:
            return float(str(text or "").strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a valid number.") from exc

    @classmethod
    def _parse_pct(cls, text: Optional[str], label: str) -> float:
        return cls._parse_float(text, label) / 100.0

    @staticmethod
    def _describe_model(model: AssetPricingModel) -> str:
        if isinstance(model, LinearFactorModel):
            return "Linear factor model: E[R_i] = R_f + Σ_k β_i,k · λ_k + α_i"
        return model.label

    @staticmethod
    def _loading_header(model: AssetPricingModel, factor_key: str) -> str:
        if len(model.factors) == 1:
            return "Beta"
        return f"β {factor_key}"

    @staticmethod
    def _default_pricing_state(model: AssetPricingModel, n: int) -> dict:
        return {
            "premia": {f.key: f"{f.default_premium * 100.0:.2f}" for f in model.factors},
            "loadings": {f.key: [f"{f.default_loading:.2f}"] * n for f in model.factors},
            "alphas": ["0.00"] * n,
        }

    def _capture_pricing_state(self) -> None:
        """Store current pricing panel text values under the active model key."""
        if not self._pricing_built:
            return
        self._pricing_state[self._pricing_model_key] = {
            "premia": {k: f.value or "" for k, f in self._pricing_premia_fields.items()},
            "loadings": {
                k: [f.value or "" for f in fields]
                for k, fields in self._pricing_loading_fields.items()
            },
            "alphas": [f.value or "" for f in self._pricing_alpha_fields],
        }

    def _rebuild_pricing_panel(self) -> None:
        model = get_pricing_model(self._pricing_model_key)
        names = [(tf.value or "").strip() for tf in self._name_fields]
        n = len(names)
        state = self._pricing_state.get(model.key) or self._default_pricing_state(model, n)

        self._pricing_model_desc.value = self._describe_model(model)

        self._pricing_premia_fields = {}
        premia_controls: list[ft.Control] = []
        for spec in model.factors:
            value = state["premia"].get(spec.key, f"{spec.default_premium * 100.0:.2f}")
            tf = ft.TextField(
                label=f"{spec.label} (%)",
                value=value,
                width=300,
                dense=True,
                on_change=self._on_pricing_input_changed,
            )
            self._pricing_premia_fields[spec.key] = tf
            premia_controls.append(tf)
        self._pricing_premia_row.controls = premia_controls

        self._pricing_loading_fields = {spec.key: [] for spec in model.factors}
        self._pricing_alpha_fields = []
        self._pricing_result_texts = []
        self._pricing_name_texts = []

        header_cells: list[ft.Control] = [ft.Text("Market", weight=ft.FontWeight.BOLD, size=12, width=160)]
        for spec in model.factors:
            header_cells.append(
                ft.Text(self._loading_header(model, spec.key), weight=ft.FontWeight.BOLD, size=12, width=110)
            )
        header_cells.append(ft.Text("Alpha (%)", weight=ft.FontWeight.BOLD, size=12, width=100))
        header_cells.append(ft.Text("E[R] (%)", weight=ft.FontWeight.BOLD, size=12, width=100))
        rows: list[ft.Control] = [ft.Row(controls=header_cells, spacing=8)]

        for i, name in enumerate(names):
            name_text = ft.Text(name, size=13, width=160)
            row_controls: list[ft.Control] = [name_text]
            self._pricing_name_texts.append(name_text)

            for spec in model.factors:
                values = state["loadings"].get(spec.key, [])
                val = values[i] if i < len(values) else f"{spec.default_loading:.2f}"
                tf = ft.TextField(
                    value=val,
                    width=110,
                    dense=True,
                    content_padding=8,
                    on_change=self._on_pricing_input_changed,
                )
                self._pricing_loading_fields[spec.key].append(tf)
                row_controls.append(tf)

            alpha_values = state["alphas"]
            alpha_val = alpha_values[i] if i < len(alpha_values) else "0.00"
            alpha_tf = ft.TextField(
                value=alpha_val,
                width=100,
                dense=True,
                content_padding=8,
                on_change=self._on_pricing_input_changed,
            )
            self._pricing_alpha_fields.append(alpha_tf)
            row_controls.append(alpha_tf)

            result_text = ft.Text("—", size=13, weight=ft.FontWeight.BOLD, width=100)
            self._pricing_result_texts.append(result_text)
            row_controls.append(result_text)

            rows.append(ft.Row(controls=row_controls, spacing=8))

        self._pricing_table_col.controls = rows
        self._pricing_built = True
        self._recompute_pricing()

    def _on_pricing_model_changed(self, _e: ft.ControlEvent) -> None:
        self._capture_pricing_state()
        self._pricing_model_key = self._pricing_model_dropdown.value or GLOBAL_CAPM_KEY
        self._rebuild_pricing_panel()
        self._safe_update()

    def _on_pricing_input_changed(self, _e: ft.ControlEvent) -> None:
        self._recompute_pricing()
        self._safe_update()

    def _on_pricing_rf_changed(self, _e: ft.ControlEvent) -> None:
        if not self._syncing_rf:
            self._syncing_rf = True
            try:
                self._rf_field.value = self._pricing_rf_field.value
            finally:
                self._syncing_rf = False
        self._recompute_pricing()
        self._safe_update()

    def _on_optimizer_rf_changed(self, _e: ft.ControlEvent) -> None:
        if not self._syncing_rf:
            self._syncing_rf = True
            try:
                self._pricing_rf_field.value = self._rf_field.value
            finally:
                self._syncing_rf = False
        self._recompute_pricing()
        self._safe_update()

    def _read_pricing_inputs(self, model: AssetPricingModel) -> PricingInputs:
        rf = self._parse_pct(self._pricing_rf_field.value, "Risk-Free Rate")
        premia = {
            key: self._parse_pct(f.value, "Risk premium")
            for key, f in self._pricing_premia_fields.items()
        }

        loadings: Optional[np.ndarray] = None
        if isinstance(model, LinearFactorModel) and model.factors:
            columns = [
                [self._parse_float(f.value, f"Loading ({spec.key})") for f in self._pricing_loading_fields[spec.key]]
                for spec in model.factors
            ]
            loadings = np.column_stack([np.asarray(c, dtype=float) for c in columns])

        alphas = np.array(
            [self._parse_pct(f.value, "Alpha") for f in self._pricing_alpha_fields],
            dtype=float,
        )
        return PricingInputs(
            risk_free_rate=rf,
            factor_premia=premia,
            loadings=loadings,
            alphas=alphas,
        )

    def _recompute_pricing(self) -> None:
        """Recompute the active pricing model, refresh the panel, and feed forward if live-sync is on."""
        if not self._pricing_built:
            return
        model = get_pricing_model(self._pricing_model_key)
        try:
            inputs = self._read_pricing_inputs(model)
            result = model.price(inputs)
        except ValueError as exc:
            self._pricing_status.value = str(exc)
            for t in self._pricing_result_texts:
                t.value = "—"
            self._pricing_last_result = None
            self._pricing_chart_image.src = b""
            self._pricing_chart_note.value = ""
            return

        self._pricing_status.value = ""
        self._pricing_last_result = result
        for t, r in zip(self._pricing_result_texts, result.expected_returns):
            t.value = f"{r * 100.0:.2f}"

        names = [t.value or "" for t in self._pricing_name_texts]
        primary_premium = (
            inputs.factor_premia[model.factors[0].key] if model.factors else None
        )
        png = render_pricing_chart_png(
            names,
            result,
            inputs.loadings,
            inputs.risk_free_rate,
            model.label,
            primary_premium,
        )
        if png is not None:
            self._pricing_chart_image.src = png
            self._pricing_chart_note.value = ""
        else:
            self._pricing_chart_image.src = b""
            self._pricing_chart_note.value = "This model does not provide a return decomposition; no chart."

        if bool(self._live_sync_cb.value):
            self._push_returns_to_optimizer(result.expected_returns)

    def _push_returns_to_optimizer(self, returns: np.ndarray) -> None:
        for tf, r in zip(self._return_fields, returns):
            tf.value = f"{float(r) * 100.0:.2f}"

    def _on_live_sync_changed(self, _e: ft.ControlEvent) -> None:
        enabled = bool(self._live_sync_cb.value)
        for tf in self._return_fields:
            tf.read_only = enabled
        if enabled and self._pricing_last_result is not None:
            self._push_returns_to_optimizer(self._pricing_last_result.expected_returns)
        self._safe_update()

    def _read_capm_extras(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Betas, alphas and ERP from the Global CAPM pricing state (for CSV save)."""
        self._capture_pricing_state()
        state = self._pricing_state.get(GLOBAL_CAPM_KEY)
        if state is None:
            return None, None, DEFAULT_ERP
        betas = np.array(
            [self._parse_float(v, "Beta") for v in state["loadings"][CAPM_FACTOR_KEY]],
            dtype=float,
        )
        alphas = np.array(
            [self._parse_pct(v, "Alpha") for v in state["alphas"]],
            dtype=float,
        )
        erp = self._parse_pct(state["premia"][CAPM_FACTOR_KEY], "Equity risk premium")
        return betas, alphas, erp

    # ------------------------------------------------------------------
    # Portfolio Optimizer tab
    # ------------------------------------------------------------------

    def _parse_inputs(self, include_pricing: bool = False) -> tuple[MarketDataset, np.ndarray]:
        names = [(tf.value or "").strip() for tf in self._name_fields]
        if any(not name for name in names):
            raise ValueError("All market names must be non-empty.")

        try:
            returns = np.array(
                [float(tf.value or "") / 100.0 for tf in self._return_fields],
                dtype=float,
            )
        except ValueError as exc:
            raise ValueError("Expected returns must be valid numbers.") from exc

        try:
            raw_risk = np.array(
                [float(tf.value or "") for tf in self._risk_fields], dtype=float
            )
        except ValueError as exc:
            raise ValueError("Volatility / Variance values must be valid numbers.") from exc

        if (self._risk_mode_group.value or "volatility") == "volatility":
            volatilities = raw_risk / 100.0
            if np.any(volatilities < 0.0):
                raise ValueError("Volatilities must be non-negative.")
        else:
            if np.any(raw_risk < 0.0):
                raise ValueError("Variances must be non-negative.")
            volatilities = np.sqrt(raw_risk)

        n = len(names)
        corr = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        corr[i, j] = float(self._corr_fields[i][j].value or "")
                    except ValueError as exc:
                        raise ValueError(
                            f"Correlation at ({names[i]}, {names[j]}) must be numeric."
                        ) from exc

        try:
            rf = float(self._rf_field.value or "") / 100.0
        except ValueError as exc:
            raise ValueError("Risk-Free Rate must be a valid number.") from exc

        betas: Optional[np.ndarray] = None
        alphas: Optional[np.ndarray] = None
        erp = DEFAULT_ERP
        if include_pricing:
            betas, alphas, erp = self._read_capm_extras()

        cov = build_covariance_matrix(corr, volatilities=volatilities)
        dataset = MarketDataset(
            names=names,
            expected_returns=returns,
            volatilities=volatilities,
            corr_matrix=corr,
            risk_free_rate=rf,
            betas=betas,
            alphas=alphas,
            erp=erp,
        )
        return dataset, cov

    def run_optimization(self, update_page: bool = True) -> None:
        """Run portfolio optimization using src.optimizer and update chart & summary."""
        try:
            dataset, cov = self._parse_inputs()
            allow_short = bool(self._allow_short_cb.value)
            mvp = find_min_variance_portfolio(cov, allow_short=allow_short)
            tangency = find_tangency_portfolio(
                dataset.expected_returns,
                cov,
                dataset.risk_free_rate,
                allow_short=allow_short,
            )
            frontier = compute_efficient_frontier(
                dataset.expected_returns,
                cov,
                num_points=50,
                allow_short=allow_short,
            )

            cal_result: Optional[CALPortfolioResult] = None
            cal_mode = self._cal_mode_group.value or "none"
            if cal_mode in ("return", "volatility"):
                try:
                    target_val = float(self._cal_target_field.value or "") / 100.0
                except ValueError as exc:
                    raise ValueError("CAL target value must be a valid number.") from exc
                if cal_mode == "return":
                    cal_result = find_cal_portfolio(
                        tangency, dataset.risk_free_rate, target_return=target_val
                    )
                else:
                    cal_result = find_cal_portfolio(
                        tangency, dataset.risk_free_rate, target_volatility=target_val
                    )
        except Exception as exc:
            self._show_dialog("Optimization Error", str(exc))
            return

        self._last_names = dataset.names
        self._last_tangency = tangency
        self._last_cal = cal_result

        self._chart_image.src = render_frontier_chart_png(
            dataset, mvp, tangency, frontier, cal_result
        )
        self._update_summary(dataset, mvp, tangency, cal_result)

        if update_page:
            self._safe_update()

        if not allow_short and dataset.risk_free_rate >= float(
            np.max(dataset.expected_returns)
        ):
            self._show_dialog(
                "Risk-Free Rate Notice",
                "The risk-free rate is greater than or equal to all expected market "
                "returns. In long-only mode, no risky portfolio beats the risk-free "
                "asset, so the Tangency Portfolio has a non-positive Sharpe ratio.",
            )

    def _update_summary(
        self,
        dataset: MarketDataset,
        mvp: PortfolioResult,
        tangency: PortfolioResult,
        cal_result: Optional[CALPortfolioResult] = None,
    ) -> None:
        self._weights_table.rows.clear()

        if cal_result is not None:
            exp_ret = cal_result.expected_return * 100.0
            vol = cal_result.volatility * 100.0
            sharpe = cal_result.sharpe_ratio
            y_pct = cal_result.risky_allocation * 100.0
            rf_pct = cal_result.risk_free_weight * 100.0
            self._metrics_text.value = (
                f"CAL Target — Return: {exp_ret:.2f}%  |  Vol: {vol:.2f}%  |  "
                f"Sharpe: {sharpe:.3f}  |  Risky: {y_pct:.1f}% / Rf: {rf_pct:.1f}%"
            )
            for name, weight in zip(dataset.names, cal_result.risky_weights):
                self._weights_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(name)),
                            ft.DataCell(ft.Text(f"{weight * 100.0:.2f}%")),
                        ]
                    )
                )
            self._weights_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text("Risk-Free Asset", weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Text(f"{rf_pct:.2f}%", weight=ft.FontWeight.BOLD)),
                    ]
                )
            )
        else:
            exp_ret = (tangency.expected_return or 0.0) * 100.0
            vol = tangency.volatility * 100.0
            sharpe = tangency.sharpe_ratio or 0.0
            mvp_vol = mvp.volatility * 100.0
            self._metrics_text.value = (
                f"Expected Return: {exp_ret:.2f}%  |  Volatility: {vol:.2f}%  |  "
                f"Sharpe Ratio: {sharpe:.3f}  |  MVP Vol: {mvp_vol:.2f}%"
            )
            for name, weight in zip(dataset.names, tangency.weights):
                self._weights_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(name)),
                            ft.DataCell(ft.Text(f"{weight * 100.0:.2f}%")),
                        ]
                    )
                )

    # ------------------------------------------------------------------
    # Dialogs & file I/O
    # ------------------------------------------------------------------

    def _on_load_preset(self) -> None:
        self.load_dataset(get_default_preset())
        self.run_optimization()

    def _show_dialog(self, title: str, message: str) -> None:
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Text(message),
            actions=[
                ft.TextButton("OK", on_click=lambda _: self.page.pop_dialog()),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dlg)

    def _show_snackbar(self, message: str) -> None:
        self.page.show_dialog(ft.SnackBar(content=ft.Text(message), open=True))

    async def _on_import_csv(self, _e: ft.ControlEvent) -> None:
        try:
            files = await self.file_picker.pick_files(
                dialog_title="Import Market Assumptions CSV",
                allowed_extensions=["csv"],
                allow_multiple=False,
                with_data=True,
            )
            if not files:
                return
            picked = files[0]
            if picked.bytes is not None:
                csv_text = bytes(picked.bytes).decode("utf-8")
            elif picked.path:
                csv_text = Path(picked.path).read_text(encoding="utf-8")
            else:
                raise ValueError("Unable to read selected CSV file.")

            rf = float(self._rf_field.value or "3.0") / 100.0
            _, _, current_erp = self._safe_capm_extras()
            dataset = parse_market_assumptions_csv_text(
                csv_text, risk_free_rate=rf, erp=current_erp
            )
            self.load_dataset(dataset)
            self.run_optimization()
            self._show_snackbar(f"Imported {picked.name}")
        except Exception as exc:
            self._show_dialog("Import Error", str(exc))

    def _safe_capm_extras(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        try:
            return self._read_capm_extras()
        except ValueError:
            return None, None, DEFAULT_ERP

    async def _on_save_csv(self, _e: ft.ControlEvent) -> None:
        try:
            dataset, _cov = self._parse_inputs(include_pricing=True)
            csv_text = serialize_market_assumptions_csv(dataset)
            saved_path = await self.file_picker.save_file(
                dialog_title="Save Market Assumptions CSV",
                file_name="market_assumptions.csv",
                allowed_extensions=["csv"],
                src_bytes=csv_text.encode("utf-8"),
            )
            if saved_path is not None:
                self._show_snackbar("Market assumptions CSV saved.")
        except Exception as exc:
            self._show_dialog("Save Error", str(exc))

    async def _on_export_weights_csv(self, _e: ft.ControlEvent) -> None:
        if self._last_tangency is None or not self._last_names:
            self._show_dialog(
                "No Results", "Run an optimization before exporting portfolio weights."
            )
            return
        try:
            if self._last_cal is not None:
                export_names = [*self._last_names, "Risk-Free Asset"]
                export_weights = np.append(
                    self._last_cal.risky_weights, self._last_cal.risk_free_weight
                )
            else:
                export_names = self._last_names
                export_weights = self._last_tangency.weights
            csv_text = serialize_weights_csv(export_names, export_weights)
            saved_path = await self.file_picker.save_file(
                dialog_title="Export Optimal Weights to CSV",
                file_name="optimal_weights.csv",
                allowed_extensions=["csv"],
                src_bytes=csv_text.encode("utf-8"),
            )
            if saved_path is not None:
                self._show_snackbar("Optimal weights CSV exported.")
        except Exception as exc:
            self._show_dialog("Export Error", str(exc))


def main(page: ft.Page) -> None:
    """Flet application entry point."""
    PortfolioOptimizerView(page)
