"""Flet v1.0+ UI and Matplotlib visualization for the Portfolio Optimizer.

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

    # Efficient frontier curve
    ax.plot(
        frontier.volatilities * 100.0,
        frontier.returns * 100.0,
        color="#1f77b4",
        linewidth=2.2,
        label="Efficient Frontier",
    )

    # Individual markets
    vols_pct = dataset.volatilities * 100.0
    rets_pct = dataset.expected_returns * 100.0
    ax.scatter(
        vols_pct,
        rets_pct,
        color="#555555",
        s=45,
        zorder=4,
        label="Individual Markets",
    )
    for name, x, y in zip(dataset.names, vols_pct, rets_pct):
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
        )

    # Minimum Variance Portfolio (MVP)
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

    # Tangency Portfolio (Max Sharpe)
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

    # Capital Allocation Line (CAL)
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


class PortfolioOptimizerView:
    """Flet UI controller for the International Equity Portfolio Optimizer."""

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._syncing_corr = False

        self._name_fields: list[ft.TextField] = []
        self._return_fields: list[ft.TextField] = []
        self._risk_fields: list[ft.TextField] = []
        self._corr_fields: list[list[ft.TextField]] = []
        self._corr_col_labels: list[ft.Text] = []
        self._corr_row_labels: list[ft.Text] = []

        self._last_names: list[str] = []
        self._last_tangency: Optional[PortfolioResult] = None
        self._last_cal: Optional[CALPortfolioResult] = None

        self.file_picker = ft.FilePicker()
        self.page.services.append(self.file_picker)

        self._configure_page()
        self._build_layout()
        self.load_dataset(get_default_preset())
        self.run_optimization(update_page=False)
        self.page.update()

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

    def _build_layout(self) -> None:
        # Risk input mode selector
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

        # Correlation matrix container
        self._corr_grid_col = ft.Column(spacing=4, tight=True)

        # Optimization parameters & CAL target controls
        self._rf_field = ft.TextField(
            label="Risk-Free Rate (%)",
            value="3.00",
            width=160,
            dense=True,
        )
        self._allow_short_cb = ft.Checkbox(
            label="Allow Short Selling",
            value=False,
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

        # Right panel: Chart + Summary
        self._chart_image = ft.Image(
            src=b"",
            fit=ft.BoxFit.CONTAIN,
            height=420,
        )

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

        header = ft.Row(
            controls=[
                ft.Text(
                    "International Equity Portfolio Optimizer",
                    size=22,
                    weight=ft.FontWeight.BOLD,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
        )

        self.page.add(
            header,
            ft.ResponsiveRow(
                controls=[left_col, right_col],
                spacing=14,
                run_spacing=14,
            ),
        )

    @staticmethod
    def _short_label(name: str, max_len: int = 8) -> str:
        clean = name.strip()
        return clean if len(clean) <= max_len else clean[: max_len - 1] + "…"

    def load_dataset(self, dataset: MarketDataset) -> None:
        """Populate input fields and correlation matrix from a MarketDataset."""
        self._rf_field.value = f"{dataset.risk_free_rate * 100.0:.2f}"
        self._build_markets_table(dataset)
        self._build_corr_grid(dataset)

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
            short = self._short_label(self._name_fields[idx].value or "")
            self._corr_col_labels[idx].value = short
            self._corr_row_labels[idx].value = short
            self.page.update()

    def _mirror_corr_cell(self, row: int, col: int) -> None:
        """Auto-mirror edits between upper and lower triangle of the correlation grid."""
        if self._syncing_corr or row == col:
            return
        self._syncing_corr = True
        try:
            val = self._corr_fields[row][col].value
            if self._corr_fields[col][row].value != val:
                self._corr_fields[col][row].value = val
                self.page.update()
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
                    vol = val / 100.0
                    tf.value = f"{vol ** 2:.6f}"
                else:
                    if val >= 0.0:
                        tf.value = f"{np.sqrt(val) * 100.0:.2f}"
            except ValueError:
                continue
        self.page.update()

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
        self.page.update()

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

    def _parse_inputs(self) -> tuple[MarketDataset, np.ndarray]:
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
                if i == j:
                    corr[i, j] = 1.0
                else:
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

        cov = build_covariance_matrix(corr, volatilities=volatilities)
        dataset = MarketDataset(
            names=names,
            expected_returns=returns,
            volatilities=volatilities,
            corr_matrix=corr,
            risk_free_rate=rf,
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
            self.page.update()

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
                        ft.DataCell(
                            ft.Text("Risk-Free Asset", weight=ft.FontWeight.BOLD)
                        ),
                        ft.DataCell(
                            ft.Text(f"{rf_pct:.2f}%", weight=ft.FontWeight.BOLD)
                        ),
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

    def _on_load_preset(self) -> None:
        self.load_dataset(get_default_preset())
        self.run_optimization()

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
            dataset = parse_market_assumptions_csv_text(csv_text, risk_free_rate=rf)
            self.load_dataset(dataset)
            self.run_optimization()
            self._show_snackbar(f"Imported {picked.name}")
        except Exception as exc:
            self._show_dialog("Import Error", str(exc))

    async def _on_save_csv(self, _e: ft.ControlEvent) -> None:
        try:
            dataset, _cov = self._parse_inputs()
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
