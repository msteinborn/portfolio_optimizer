"""Pure mathematical engine for Mean-Variance Portfolio Optimization.

Zero GUI/Tkinter dependencies. All inputs use decimal returns/volatilities
(e.g., 0.08 for 8%, 0.15 for 15%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

PSD_TOLERANCE = 1e-8
SYMMETRY_TOLERANCE = 1e-8
DIAGONAL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PortfolioResult:
    """Result of a single-portfolio optimization."""

    weights: np.ndarray
    volatility: float
    expected_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None


@dataclass(frozen=True)
class EfficientFrontierResult:
    """Result of an efficient frontier sweep."""

    returns: np.ndarray
    volatilities: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class CALPortfolioResult:
    """Complete portfolio on the Capital Allocation Line (Tangency + Risk-Free)."""

    risky_weights: np.ndarray
    risk_free_weight: float
    risky_allocation: float
    expected_return: float
    volatility: float
    sharpe_ratio: float


def is_positive_semidefinite(matrix: np.ndarray, tol: float = PSD_TOLERANCE) -> bool:
    """Return True if a symmetric matrix is positive semi-definite within tolerance."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    sym = 0.5 * (arr + arr.T)
    eigenvalues = np.linalg.eigvalsh(sym)
    return bool(np.min(eigenvalues) >= -tol)


def validate_correlation_matrix(corr_matrix: np.ndarray) -> np.ndarray:
    """Validate and return a clean float64 correlation matrix.

    Checks:
    - 2D square matrix with at least 1 asset
    - All entries finite
    - Symmetric within tolerance
    - Diagonal entries equal to 1.0 within tolerance
    - All entries in [-1.0, 1.0]
    - Positive semi-definite
    """
    corr = np.asarray(corr_matrix, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] == 0:
        raise ValueError("Correlation matrix must be a non-empty square 2D array.")
    if not np.all(np.isfinite(corr)):
        raise ValueError("Correlation matrix contains non-finite values.")
    if not np.allclose(corr, corr.T, atol=SYMMETRY_TOLERANCE):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0, atol=DIAGONAL_TOLERANCE):
        raise ValueError("Correlation matrix diagonal entries must all equal 1.0.")
    if np.any(corr < -1.0 - 1e-8) or np.any(corr > 1.0 + 1e-8):
        raise ValueError("Correlation values must be between -1.0 and 1.0.")

    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    if not is_positive_semidefinite(corr):
        raise ValueError("Correlation matrix is not positive semi-definite (PSD).")
    return corr


def validate_covariance_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    """Validate and return a clean symmetric float64 covariance matrix."""
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("Covariance matrix must be a non-empty square 2D array.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix contains non-finite values.")
    if not np.allclose(cov, cov.T, atol=SYMMETRY_TOLERANCE):
        raise ValueError("Covariance matrix must be symmetric.")

    cov = 0.5 * (cov + cov.T)
    if not is_positive_semidefinite(cov):
        raise ValueError("Covariance matrix is not positive semi-definite (PSD).")
    return cov


def build_covariance_matrix(
    corr_matrix: np.ndarray,
    *,
    volatilities: Optional[np.ndarray] = None,
    variances: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build covariance matrix Sigma = D @ R @ D from volatilities or variances and R.

    Exactly one of `volatilities` or `variances` must be provided.
    """
    if (volatilities is None) == (variances is None):
        raise ValueError("Provide exactly one of `volatilities` or `variances`.")

    corr = validate_correlation_matrix(corr_matrix)
    n = corr.shape[0]

    if volatilities is not None:
        vols = np.asarray(volatilities, dtype=float)
        if vols.ndim != 1 or vols.shape[0] != n:
            raise ValueError(
                f"Volatilities length ({vols.shape}) must match correlation matrix size ({n})."
            )
        if not np.all(np.isfinite(vols)) or np.any(vols < 0.0):
            raise ValueError("Volatilities must be finite and non-negative.")
    else:
        vars_arr = np.asarray(variances, dtype=float)
        if vars_arr.ndim != 1 or vars_arr.shape[0] != n:
            raise ValueError(
                f"Variances length ({vars_arr.shape}) must match correlation matrix size ({n})."
            )
        if not np.all(np.isfinite(vars_arr)) or np.any(vars_arr < 0.0):
            raise ValueError("Variances must be finite and non-negative.")
        vols = np.sqrt(vars_arr)

    cov = np.outer(vols, vols) * corr
    cov = 0.5 * (cov + cov.T)

    if not is_positive_semidefinite(cov):
        raise ValueError("Constructed covariance matrix is not positive semi-definite.")
    return cov


def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """Compute expected portfolio return: w^T mu."""
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(returns, dtype=float)
    if w.ndim != 1 or mu.ndim != 1 or w.shape[0] != mu.shape[0]:
        raise ValueError("Weights and returns must be 1D arrays of equal length.")
    return float(np.dot(w, mu))


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio variance: w^T Sigma w (clamped at 0 for numerical noise)."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    if w.ndim != 1 or cov.ndim != 2 or cov.shape != (w.shape[0], w.shape[0]):
        raise ValueError("Weights length must match covariance matrix dimensions.")
    var = float(w @ cov @ w)
    return max(0.0, var)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio volatility (standard deviation): sqrt(w^T Sigma w)."""
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))


def portfolio_sharpe_ratio(
    weights: np.ndarray,
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Compute portfolio Sharpe ratio: (E[R_p] - R_f) / sigma_p."""
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol <= 1e-12:
        return 0.0
    return float((ret - float(risk_free_rate)) / vol)


def find_min_variance_portfolio(
    cov_matrix: np.ndarray,
    allow_short: bool = False,
) -> PortfolioResult:
    """Find the Minimum Variance Portfolio (MVP).

    Minimizes w^T Sigma w subject to sum(w) = 1 (and w_i >= 0 if allow_short is False).
    """
    cov = validate_covariance_matrix(cov_matrix)
    n = cov.shape[0]
    w0 = np.full(n, 1.0 / n, dtype=float)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if allow_short else [(0.0, 1.0)] * n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum variance optimization failed: {result.message}")

    weights = result.x
    if not allow_short:
        weights = np.clip(weights, 0.0, 1.0)
        weights = weights / np.sum(weights)

    vol = portfolio_volatility(weights, cov)
    return PortfolioResult(weights=weights, volatility=vol)


def find_tangency_portfolio(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    allow_short: bool = False,
) -> PortfolioResult:
    """Find the Tangency Portfolio maximizing the Sharpe ratio.

    Maximizes (w^T mu - R_f) / sqrt(w^T Sigma w) subject to sum(w) = 1
    (and w_i >= 0 if allow_short is False).
    """
    mu = np.asarray(returns, dtype=float)
    cov = validate_covariance_matrix(cov_matrix)
    n = cov.shape[0]
    if mu.ndim != 1 or mu.shape[0] != n:
        raise ValueError("Returns length must match covariance matrix dimensions.")
    if not np.all(np.isfinite(mu)):
        raise ValueError("Returns must be finite.")

    rf = float(risk_free_rate)

    def neg_sharpe(w: np.ndarray) -> float:
        var = max(float(w @ cov @ w), 1e-16)
        vol = np.sqrt(var)
        return -float((np.dot(w, mu) - rf) / vol)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if allow_short else [(0.0, 1.0)] * n

    # Try multiple starting points (equal weight + MVP) for robustness
    mvp = find_min_variance_portfolio(cov, allow_short=allow_short)
    starts = [np.full(n, 1.0 / n, dtype=float), mvp.weights]

    best_result = None
    for w0 in starts:
        res = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success and (best_result is None or res.fun < best_result.fun):
            best_result = res

    if best_result is None:
        raise RuntimeError("Tangency portfolio optimization failed to converge.")

    weights = best_result.x
    if not allow_short:
        weights = np.clip(weights, 0.0, 1.0)
        weights = weights / np.sum(weights)

    exp_ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    sharpe = portfolio_sharpe_ratio(weights, mu, cov, rf)

    return PortfolioResult(
        weights=weights,
        volatility=vol,
        expected_return=exp_ret,
        sharpe_ratio=sharpe,
    )


def compute_efficient_frontier(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_points: int = 50,
    allow_short: bool = False,
) -> EfficientFrontierResult:
    """Compute the Mean-Variance Efficient Frontier.

    Sweeps target returns from the MVP expected return up to the maximum
    achievable return (max(mu) for long-only; extended range when shorting is allowed).
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    mu = np.asarray(returns, dtype=float)
    cov = validate_covariance_matrix(cov_matrix)
    n = cov.shape[0]
    if mu.ndim != 1 or mu.shape[0] != n:
        raise ValueError("Returns length must match covariance matrix dimensions.")
    if not np.all(np.isfinite(mu)):
        raise ValueError("Returns must be finite.")

    mvp = find_min_variance_portfolio(cov, allow_short=allow_short)
    mvp_return = portfolio_return(mvp.weights, mu)

    max_return = float(np.max(mu))
    if allow_short:
        span = max(max_return - mvp_return, 0.02)
        upper_return = max_return + 0.5 * span
    else:
        upper_return = max(max_return, mvp_return)

    target_returns = np.linspace(mvp_return, upper_return, num_points)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    bounds = None if allow_short else [(0.0, 1.0)] * n

    frontier_returns: list[float] = []
    frontier_vols: list[float] = []
    frontier_weights: list[np.ndarray] = []

    w_prev = mvp.weights.copy()

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(np.dot(w, mu) - t)},
        ]
        res = minimize(
            objective,
            w_prev,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            # Retry from equal weights if warm-start failed
            res = minimize(
                objective,
                np.full(n, 1.0 / n, dtype=float),
                method="SLSQP",
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
        if res.success:
            w = res.x
            if not allow_short:
                w = np.clip(w, 0.0, 1.0)
                w = w / np.sum(w)
            w_prev = w
            frontier_returns.append(portfolio_return(w, mu))
            frontier_vols.append(portfolio_volatility(w, cov))
            frontier_weights.append(w)

    if len(frontier_returns) == 0:
        raise RuntimeError("Efficient frontier optimization failed for all target returns.")

    return EfficientFrontierResult(
        returns=np.array(frontier_returns, dtype=float),
        volatilities=np.array(frontier_vols, dtype=float),
        weights=np.array(frontier_weights, dtype=float),
    )


def find_cal_portfolio(
    tangency: PortfolioResult,
    risk_free_rate: float,
    *,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
) -> CALPortfolioResult:
    """Compute a complete portfolio on the Capital Allocation Line (CAL).

    Exactly one of `target_return` or `target_volatility` must be provided.
    Combines the Tangency Portfolio (weight y) with the risk-free asset (weight 1 - y).
    """
    if (target_return is None) == (target_volatility is None):
        raise ValueError(
            "Provide exactly one of `target_return` or `target_volatility`."
        )

    if tangency.expected_return is None:
        raise ValueError("Tangency portfolio must include `expected_return`.")
    if tangency.volatility <= 1e-12:
        raise ValueError("Tangency portfolio volatility must be positive.")

    rf = float(risk_free_rate)
    tang_ret = float(tangency.expected_return)
    tang_vol = float(tangency.volatility)

    if tang_ret <= rf + 1e-12:
        raise ValueError(
            "Cannot build a positive-slope CAL when Tangency Portfolio expected "
            "return does not exceed the risk-free rate."
        )

    if target_return is not None:
        t_ret = float(target_return)
        if not np.isfinite(t_ret):
            raise ValueError("Target return must be finite.")
        y = (t_ret - rf) / (tang_ret - rf)
        exp_ret = t_ret
        vol = float(abs(y) * tang_vol)
    else:
        assert target_volatility is not None
        t_vol = float(target_volatility)
        if not np.isfinite(t_vol) or t_vol < 0.0:
            raise ValueError("Target volatility must be finite and non-negative.")
        y = t_vol / tang_vol
        vol = t_vol
        exp_ret = float(rf + y * (tang_ret - rf))

    risky_weights = y * np.asarray(tangency.weights, dtype=float)
    rf_weight = float(1.0 - y)
    sharpe = 0.0 if vol <= 1e-12 else float((exp_ret - rf) / vol)

    return CALPortfolioResult(
        risky_weights=risky_weights,
        risk_free_weight=rf_weight,
        risky_allocation=float(y),
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
    )
