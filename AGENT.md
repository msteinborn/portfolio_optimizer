# AGENT.md — International Equity Portfolio Optimizer

## Project Overview
Lightweight Python desktop application that computes the Mean-Variance Efficient Frontier for international equity markets, identifies the Minimum Variance Portfolio (MVP) and Tangency Portfolio (Max Sharpe Ratio), plots the Capital Allocation Line (CAL), and solves for target-return or target-volatility portfolios along the CAL.

## Tech Stack
- **Language:** Python 3.10+ (`.venv` managed via `uv`, dependencies in `pyproject.toml`)
- **Math & Optimization:** `numpy`, `scipy` (`scipy.optimize.minimize`), stdlib `csv` (no `pandas`)
- **Plotting & UI:** `matplotlib` (`agg` backend rendered to PNG bytes) in `flet` v1.0+ (`ft.run(main)`)
- **Deployment:** Static Pyodide web app via `flet publish main.py` (`./dist`)
- **Testing:** `pytest` (configured in `pyproject.toml`)

## Architecture
Keep business logic strictly separated from the GUI:
- `src/optimizer.py` — Pure math engine (zero UI imports): covariance builder & PSD validation, portfolio metrics, MVP, Tangency Portfolio, Efficient Frontier, and CAL target portfolio solver.
- `src/data.py` — `MarketDataset`, default 5-market international equity preset, stdlib `csv` + `io.StringIO` import/export for market assumptions and optimal weights.
- `src/ui.py` — `PortfolioOptimizerView` Flet v1.0+ UI, auto-mirroring correlation grid, CAL target selector, Matplotlib PNG chart renderer, browser-safe `FilePicker` CSV I/O, and summary table.
- `main.py` — Application entry point (`ft.run(main)`).
- `vercel.json` — Static Vercel deployment config (`framework: null`, installs `flet`/`flet-cli`/`flet-web` into `/tmp/flet-venv` via `uv` to satisfy PEP 668 and keep the venv outside the repo, runs `flet publish main.py`, serves `dist/`).
- `tests/test_optimizer.py`, `tests/test_data.py`, `tests/test_ui.py` — Unit tests (`.venv/bin/pytest -v`).

## Mathematical Specifications
- **Inputs:** $N$ markets with annualized expected returns ($\mu$), volatilities ($\sigma$) or variances ($\sigma^2$), an $N \times N$ PSD correlation matrix ($R$, giving $\Sigma = D R D$), and risk-free rate ($R_f$).
- **Constraints:** Fully invested ($\sum w_i = 1$), long-only ($w_i \ge 0$) by default with optional short-selling toggle.
- **Outputs:** Efficient Frontier curve, MVP, Tangency Portfolio (maximizes $(E[R_p] - R_f) / \sigma_p$), CAL from $(0, R_f)$ through Tangency, and optional CAL target portfolio (mutually exclusive: target return OR target volatility, combining risky weight $y$ with risk-free weight $1 - y$).

## Build Status

### Phase 1: First Build
- [x] Core math engine (`src/optimizer.py`): PSD-checked covariance builder, portfolio metrics, MVP, Tangency Portfolio, Efficient Frontier, and CAL target solver (`find_cal_portfolio`).
- [x] Data & CSV I/O (`src/data.py`): 5-market international equity preset plus CSV import/export for assumptions and optimal weights.
- [x] Desktop UI (`src/ui.py`, `main.py`): Editable assumptions table (volatility/variance toggle), auto-mirroring correlation matrix (diagonal locked to `1.0`), short-selling toggle, mutually exclusive CAL target selector, embedded Matplotlib plot, summary table, and error/warning popups.
- [x] Automated tests (`tests/`): 24 passing `pytest` tests covering math engine, CAL targets, data/CSV round-trips, and edge cases.

### Phase 2: Flet Web Migration & Static Deployment
- [x] **Trim heavy dependencies (`src/data.py`, `tests/test_data.py`):** Replace `pandas` with Python's built-in `csv` module and add in-memory string/stream CSV helpers (`io.StringIO`) so Pyodide downloads fewer wheels and browser file uploads/downloads work without local filesystem paths.
- [x] **Manage dependencies via `pyproject.toml`:** Replace `requirements.txt` with a clean `pyproject.toml` listing only hand-picked direct dependencies (`flet>=1.0`, `numpy`, `scipy`, `matplotlib`; `pytest` in dev/optional deps) so the Pyodide build does not choke on transitive or incompatible wheels.
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

## Git & Deployment Workflow
- **Remote (`origin`):** `git@github.com:msteinborn/portfolio_optimizer.git` (SSH — use SSH, not HTTPS, as this machine authenticates to GitHub via `~/.ssh/id_ed25519` with `ssh -T git@github.com`).
- **Push / Pull:** `git push origin <branch>` and `git pull origin <branch>` work directly over SSH without prompts.
- **Vercel:** Connected to `msteinborn/portfolio_optimizer`; pushing commits automatically triggers builds via `vercel.json`.

## Agent Guidelines
- Do not mix `scipy` calculations inside UI event callbacks; call `src/optimizer.py`.
- Always run `.venv/bin/pytest -v` after modifying `src/optimizer.py` or `src/data.py`.
- Keep dependencies minimal and Pyodide-compatible when targeting static web deployment.
