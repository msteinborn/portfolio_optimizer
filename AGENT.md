# AGENT.md — International Equity Portfolio Optimizer

## Project Overview
Lightweight Python application (desktop via Flet, or static Pyodide web app) with two views:
1. **Asset Pricing:** Pluggable asset pricing models (starting with Global CAPM) that produce expected returns \(\mu\) from factor premia, betas, and optional alphas.
2. **Portfolio Optimizer:** Computes the Mean-Variance Efficient Frontier for international equity markets, identifies the Minimum Variance Portfolio (MVP) and Tangency Portfolio (Max Sharpe Ratio), plots the Capital Allocation Line (CAL), and solves for target-return or target-volatility portfolios along the CAL.

## Tech Stack
- **Language:** Python 3.10+ (`.venv` managed via `uv`, dependencies in `pyproject.toml`)
- **Math & Optimization:** `numpy`, `scipy` (`scipy.optimize.minimize`), stdlib `csv` (no `pandas`)
- **Plotting & UI:** `matplotlib` (`agg` backend rendered to PNG bytes) in `flet` v1.0+ (`ft.run(main)`)
- **Deployment:** Static Pyodide web app via `flet publish main.py` (`./dist`)
- **Testing:** `pytest` (configured in `pyproject.toml`)

## Architecture
Keep business logic strictly separated from the GUI:
- `src/optimizer.py` — Pure math engine (zero UI imports): covariance builder & PSD validation, portfolio metrics, MVP, Tangency Portfolio, Efficient Frontier, and CAL target portfolio solver.
- `src/pricing.py` — Pure pricing engine (zero UI imports): `AssetPricingModel` base class (`price(inputs) -> PricingResult`), `LinearFactorModel` subclass (\(E[R_i] = R_f + \sum_k \beta_{i,k}\lambda_k + \alpha_i\)), `FactorSpec`, `PricingInputs`, `PricingResult` (with optional `factor_contributions`), Global CAPM (`GLOBAL_CAPM`), and the model registry (`PRICING_MODELS`, `register_pricing_model`, `get_pricing_model`, `list_pricing_models`).
- `src/data.py` — `MarketDataset` (including optional `betas`, `alphas`, and `erp`), default 5-market international equity preset with baked-in Global CAPM betas (`DEFAULT_BETA_BY_MARKET`, `DEFAULT_ERP`), stdlib `csv` + `io.StringIO` import/export for market assumptions (optional `beta`/`alpha` columns) and optimal weights.
- `src/ui.py` — `PortfolioOptimizerView` Flet v1.0+ UI with a two-view segmented control (Asset Pricing / Portfolio Optimizer), dynamic pricing-model panel, live-sync feed-forward into the optimizer, auto-mirroring correlation grid, CAL target selector, Matplotlib PNG chart renderers (frontier, SML, factor decomposition), browser-safe `FilePicker` CSV I/O, and summary table.
- `main.py` — Application entry point (`ft.run(main)`).
- `vercel.json` — Static Vercel deployment config (`framework: null`, installs `flet`/`flet-cli`/`flet-web` into `/tmp/flet-venv` via `uv` to satisfy PEP 668 and keep the venv outside the repo, runs `flet publish main.py`, serves `dist/`).
- `tests/test_optimizer.py`, `tests/test_pricing.py`, `tests/test_data.py`, `tests/test_ui.py` — Unit tests (`.venv/bin/pytest -v`).

## Mathematical Specifications
- **Inputs:** $N$ markets with annualized expected returns ($\mu$), volatilities ($\sigma$) or variances ($\sigma^2$), an $N \times N$ PSD correlation matrix ($R$, giving $\Sigma = D R D$), and risk-free rate ($R_f$).
- **Pricing (Phase 3):** Global CAPM: $E[R_i] = R_f + \beta_i \cdot ERP + \alpha_i$, with $\alpha_i = 0$ by default and $ERP = E[R_m] - R_f$ (default 4.5%).
- **Constraints:** Fully invested ($\sum w_i = 1$), long-only ($w_i \ge 0$) by default with optional short-selling toggle.
- **Outputs:** Efficient Frontier curve, MVP, Tangency Portfolio (maximizes $(E[R_p] - R_f) / \sigma_p$), CAL from $(0, R_f)$ through Tangency, and optional CAL target portfolio (mutually exclusive: target return OR target volatility, combining risky weight $y$ with risk-free weight $1 - y$).

## Build Status

### Phase 1: First Build
- [x] Core math engine (`src/optimizer.py`): PSD-checked covariance builder, portfolio metrics, MVP, Tangency Portfolio, Efficient Frontier, and CAL target solver (`find_cal_portfolio`).
- [x] Data & CSV I/O (`src/data.py`): 5-market international equity preset plus CSV import/export for assumptions and optimal weights.
- [x] Desktop UI (`src/ui.py`, `main.py`): Editable assumptions table (volatility/variance toggle), auto-mirroring correlation matrix (diagonal locked to `1.0`), short-selling toggle, mutually exclusive CAL target selector, embedded Matplotlib plot, summary table, and error/warning popups.
- [x] Automated tests (`tests/`): passing `pytest` tests covering math engine, CAL targets, data/CSV round-trips, and edge cases.

### Phase 2: Flet Web Migration & Static Deployment
- [x] **Trim heavy dependencies (`src/data.py`, `tests/test_data.py`):** Replace `pandas` with Python's built-in `csv` module and add in-memory string/stream CSV helpers (`io.StringIO`) so Pyodide downloads fewer wheels and browser file uploads/downloads work without local filesystem paths.
- [x] **Manage dependencies via `pyproject.toml`:** Clean `pyproject.toml` listing only hand-picked direct dependencies (`flet>=1.0`, `numpy`, `scipy`, `matplotlib`; `pytest` in dev/optional deps).
- [x] **Swap UI stack to Flet (`src/ui.py`, `main.py`):**
  - [x] Replace `tkinter` / `ttk` with `flet` v1.0+ using `ft.run(main)` (since `ft.app()` is deprecated) and remove the macOS `TCL_LIBRARY`/`TK_LIBRARY` workaround.
  - [x] Replace `FigureCanvasTkAgg` with Matplotlib `agg` PNG rendering in `ft.Image` (explicitly clearing figures to avoid WASM memory leaks) for zero-plugin Pyodide compatibility.
  - [x] Preserve all Phase 1 features: editable assumptions table (volatility/variance toggle), auto-mirroring correlation matrix (diagonal locked to `1.0`), short-selling toggle, mutually exclusive CAL target selector, and complete portfolio summary table.
  - [x] Replace `messagebox` popups with Flet `ft.AlertDialog` / `ft.SnackBar` notifications, and use responsive scrollable layout containers so the app adapts cleanly to browser viewports.
- [x] **Browser-safe CSV Import/Export:** Replace `tkinter.filedialog` with Flet's `FilePicker` / in-memory browser download flow (`with_data=True` and `src_bytes`) since Pyodide cannot access the user's native filesystem directly.
- [x] **Local Pyodide static build & Vercel deployment:**
  - [x] Run `flet publish main.py` locally and serve `./dist` via `python -m http.server --directory dist`.
  - [x] Verify in a real browser that `numpy`, `scipy`, and `matplotlib` load cleanly in Pyodide, optimization runs without console errors, correlation mirroring works, and CSV export/import functions before pushing to Vercel.
  - [x] Re-run `.venv/bin/pytest -v` to confirm all math, data, and UI unit tests pass after dropping `pandas`.
  - [x] Add `vercel.json` using `/tmp/flet-venv` via `uv` and deploy live on Vercel.

### Phase 3: Two-View Pluggable Asset Pricing Engine (Global CAPM)
- [x] **Scalable pure-math asset pricing engine (`src/pricing.py`):**
  - [x] `AssetPricingModel` base class with abstract `price(inputs) -> PricingResult`, plus `expected_returns(inputs)` convenience. `LinearFactorModel` implements the additive form $E[R_i] = R_f + \sum_k \beta_{i,k}\lambda_k + \alpha_i$ using `FactorSpec` (`key`, `label`, `default_premium`, `default_loading`).
  - [x] `PricingResult` holds `expected_returns`, `risk_free_rate`, optional `factor_contributions` (None for non-additive models), and optional `alpha_component`.
  - [x] Registry: `PRICING_MODELS`, `register_pricing_model`, `get_pricing_model`, `list_pricing_models`.
  - [x] Global CAPM (`global_capm`, 1 factor: Global Equity Risk Premium, alphas default to `0.0%`) registered as the initial model, with strict input validation (finite premia/loadings/alphas/Rf, shape checks, missing factors).
- [x] **Dataset & CSV support (`src/data.py`):**
  - [x] `MarketDataset` gains optional `betas`, `alphas`, and `erp` (default 4.5%). Baked-in Global CAPM betas for the default markets (US 1.00, Europe ex-UK 0.95, Japan 0.85, UK 0.90, Emerging Markets 1.20); unknown market names fall back to beta 1.00.
  - [x] CSV gains optional `beta` and `alpha` columns. Missing columns fall back to the baked-in betas and zero alphas, so existing Phase 1/2 CSVs still load.
- [x] **Two-view Flet UI (`src/ui.py`):**
  - [x] Two views switched via a segmented control: **1. Asset Pricing** and **2. Portfolio Optimizer**. (A segmented control was used instead of `ft.Tabs`, since `TabBarView` needs a fixed height inside the scrolling page.)
  - [x] **Asset Pricing view:** Model dropdown driven by `list_pricing_models()`, shared Risk-Free Rate (base-currency note), factor premia inputs generated from `model.factors`, per-market loading/Beta and Alpha (%) inputs, live-computed E[R] (%), per-model input preservation when switching models, and a chart chosen by decomposition: Security Market Line for single-factor models, stacked factor-contribution bars for multi-factor models, and no chart for non-decomposable models.
  - [x] **Live-sync feed-forward:** The Optimizer view's `Use Pricing Model Returns (live-sync from Asset Pricing tab)` checkbox (off by default) mirrors pricing returns into the Expected Return fields and locks them; turning it off unlocks them. Risk-Free Rate stays synced across both views.
  - [x] Market names are edited in the Optimizer view and mirrored into the Asset Pricing view.
  - [x] Pricing inputs go into the CSV save path (`beta`, `alpha`, and ERP are saved; ERP is preserved on CSV import).
- [x] **Tests & Static Build Verification:**
  - [x] `tests/test_pricing.py` (33 tests): Global CAPM math, alpha tilts, decomposition, baked-in defaults, multi-factor and non-linear model extensibility, registry, and validation.
  - [x] `tests/test_data.py` and `tests/test_ui.py`: CSV beta/alpha round-trip and backward compatibility, live-sync locking/unlocking, Risk-Free Rate sync, model switching, and error handling.
  - [x] `.venv/bin/pytest -v` passes (all 92 tests), and `.venv/bin/flet publish main.py` builds cleanly.
  - [ ] Browser check of the segmented-control switch to the Optimizer view (not yet verified; the automated browser could not click the Flutter control, so this needs a manual check after deploy).

#### Phase 3 Design Notes (Non-Linear Extensibility)
- The base contract is `AssetPricingModel.price(inputs) -> PricingResult`. Do **not** require an additive loadings × premia form at the base level; `LinearFactorModel` is one subclass.
- `PricingResult.factor_contributions` is **optional** (`None` for non-additive models). The UI only renders a decomposition chart when it is present, and shows a note otherwise.
- Model inputs are currently generated from `FactorSpec`s (factor premia and per-market loadings). Non-linear models with different parameter shapes will need a richer input schema.
- Some non-linear models need historical return series for estimation. Plan for an optional time-series data source in `src/data.py` (not required for linear models in this phase).
- Downstream (`src/optimizer.py`) only consumes μ and Σ, so any model that outputs μ can feed Tab 2 unchanged.
- Phase 3 starts with CSV `beta` only (single factor). Multi-factor CSV columns (`beta_<factor>`) are not yet supported.
- The Optimizer view's expected returns still default to the Phase 1 placeholder values, not CAPM-implied returns, so the optimizer result is unchanged until the live-sync checkbox is turned on.

## Git & Deployment Workflow
- **Remote (`origin`):** `git@github.com:msteinborn/portfolio_optimizer.git` (SSH — use SSH, not HTTPS, as this machine authenticates to GitHub via `~/.ssh/id_ed25519` with `ssh -T git@github.com`).
- **Push / Pull:** `git push origin <branch>` and `git pull origin <branch>` work directly over SSH without prompts.
- **Vercel:** Connected to `msteinborn/portfolio_optimizer`; pushing commits automatically triggers builds via `vercel.json`.

## Agent Guidelines
- Do not mix `scipy` calculations inside UI event callbacks; call `src/optimizer.py`.
- Keep pricing math in `src/pricing.py` and do not put it in UI callbacks.
- Always run `.venv/bin/pytest -v` after modifying `src/optimizer.py`, `src/pricing.py`, or `src/data.py`.
- Keep dependencies minimal and Pyodide-compatible when targeting static web deployment.
- In Flet 1.1.3: `SegmentedButton.selected` must be a list (a `set` fails to serialize), and `Dropdown` uses `on_select` (not `on_change`).
