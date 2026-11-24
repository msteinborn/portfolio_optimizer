# International Equity Portfolio Optimizer

A lightweight Mean-Variance Portfolio Optimization web and desktop app built in Python with **Flet**, **NumPy**, **SciPy**, and **Matplotlib**, deployable as a static Pyodide web application on **Vercel**.

## Features
- **Mean-Variance Efficient Frontier:** Computes and plots the Efficient Frontier, Minimum Variance Portfolio (MVP), Tangency Portfolio (Max Sharpe Ratio), and Capital Allocation Line (CAL).
- **CAL Target Portfolio:** Optionally solve for a complete portfolio along the CAL by specifying either a **Target Return (%)** or **Target Annualized Volatility (%)** (mutually exclusive), showing scaled risky market weights alongside the Risk-Free Asset allocation.
- **Interactive Market Inputs:** Editable expected returns, Volatility (%) / Variance ($\sigma^2$) toggle, short-selling toggle, and an auto-mirroring correlation matrix with PSD validation.
- **CSV Import & Export:** Load/save market assumptions and export optimal portfolio weights using Python's built-in `csv` module (browser-compatible via Flet's `FilePicker`).

## Local Setup & Commands
```bash
# Create virtual environment and install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# Run unit tests
.venv/bin/pytest -v

# Run locally with Flet
.venv/bin/flet run main.py

# Build & preview the static Pyodide web bundle
.venv/bin/flet publish main.py
.venv/bin/python -m http.server --directory dist
```

## Deployment
Configured for static deployment on Vercel via `vercel.json`, which builds the Pyodide bundle into `dist/` using `uv` and `flet publish main.py`.
