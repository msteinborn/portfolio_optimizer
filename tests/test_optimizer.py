"""Unit tests for the pure math portfolio optimization engine."""

import numpy as np
import pytest

from src.optimizer import (
    build_covariance_matrix,
    compute_efficient_frontier,
    find_cal_portfolio,
    find_min_variance_portfolio,
    find_tangency_portfolio,
    is_positive_semidefinite,
    portfolio_return,
    portfolio_sharpe_ratio,
    portfolio_variance,
    portfolio_volatility,
    validate_correlation_matrix,
)


@pytest.fixture
def sample_markets():
    """5-market international equity sample dataset."""
    returns = np.array([0.085, 0.072, 0.060, 0.068, 0.095])
    volatilities = np.array([0.16, 0.18, 0.17, 0.15, 0.23])
    corr = np.array(
        [
            [1.00, 0.78, 0.55, 0.72, 0.65],
            [0.78, 1.00, 0.60, 0.85, 0.70],
            [0.55, 0.60, 1.00, 0.58, 0.52],
            [0.72, 0.85, 0.58, 1.00, 0.66],
            [0.65, 0.70, 0.52, 0.66, 1.00],
        ]
    )
    cov = build_covariance_matrix(corr, volatilities=volatilities)
    rf = 0.03
    return {
        "returns": returns,
        "volatilities": volatilities,
        "corr": corr,
        "cov": cov,
        "rf": rf,
    }


class TestCovarianceBuilderAndValidation:
    def test_build_from_volatilities_and_variances_match(self, sample_markets):
        corr = sample_markets["corr"]
        vols = sample_markets["volatilities"]
        variances = vols**2

        cov_from_vols = build_covariance_matrix(corr, volatilities=vols)
        cov_from_vars = build_covariance_matrix(corr, variances=variances)

        assert np.allclose(cov_from_vols, cov_from_vars, atol=1e-12)
        assert np.allclose(cov_from_vols, cov_from_vols.T, atol=1e-12)
        assert np.allclose(np.diag(cov_from_vols), variances, atol=1e-12)
        assert is_positive_semidefinite(cov_from_vols)

    def test_rejects_both_or_neither_vol_and_var(self, sample_markets):
        corr = sample_markets["corr"]
        vols = sample_markets["volatilities"]
        with pytest.raises(ValueError):
            build_covariance_matrix(corr)
        with pytest.raises(ValueError):
            build_covariance_matrix(corr, volatilities=vols, variances=vols**2)

    def test_rejects_non_psd_correlation_matrix(self):
        # Impossible correlations: A-B=0.9, B-C=0.9, A-C=-0.9 is not PSD
        bad_corr = np.array(
            [
                [1.0, 0.9, -0.9],
                [0.9, 1.0, 0.9],
                [-0.9, 0.9, 1.0],
            ]
        )
        assert not is_positive_semidefinite(bad_corr)
        with pytest.raises(ValueError, match="positive semi-definite"):
            validate_correlation_matrix(bad_corr)

    def test_rejects_out_of_bounds_correlation(self):
        bad_corr = np.array([[1.0, 1.2], [1.2, 1.0]])
        with pytest.raises(ValueError, match="between -1.0 and 1.0"):
            validate_correlation_matrix(bad_corr)

    def test_rejects_non_unit_diagonal(self):
        bad_corr = np.array([[0.9, 0.2], [0.2, 1.0]])
        with pytest.raises(ValueError, match="diagonal"):
            validate_correlation_matrix(bad_corr)

    def test_rejects_negative_volatility(self):
        corr = np.eye(2)
        with pytest.raises(ValueError, match="non-negative"):
            build_covariance_matrix(corr, volatilities=np.array([0.1, -0.2]))


class TestPortfolioMetrics:
    def test_two_asset_exact_metrics(self):
        returns = np.array([0.10, 0.06])
        vols = np.array([0.20, 0.10])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        cov = build_covariance_matrix(corr, volatilities=vols)
        w = np.array([0.6, 0.4])
        rf = 0.02

        expected_ret = 0.6 * 0.10 + 0.4 * 0.06
        expected_var = (
            (0.6**2) * (0.20**2)
            + (0.4**2) * (0.10**2)
            + 2 * 0.6 * 0.4 * 0.5 * 0.20 * 0.10
        )
        expected_vol = np.sqrt(expected_var)
        expected_sharpe = (expected_ret - rf) / expected_vol

        assert np.isclose(portfolio_return(w, returns), expected_ret)
        assert np.isclose(portfolio_variance(w, cov), expected_var)
        assert np.isclose(portfolio_volatility(w, cov), expected_vol)
        assert np.isclose(portfolio_sharpe_ratio(w, returns, cov, rf), expected_sharpe)


class TestMinVariancePortfolio:
    def test_long_only_mvp_constraints_and_optimality(self, sample_markets):
        cov = sample_markets["cov"]
        mvp = find_min_variance_portfolio(cov, allow_short=False)

        assert np.isclose(np.sum(mvp.weights), 1.0, atol=1e-6)
        assert np.all(mvp.weights >= -1e-8)

        # MVP must beat random long-only portfolios on volatility
        rng = np.random.default_rng(42)
        random_weights = rng.dirichlet(np.ones(len(mvp.weights)), size=2000)
        random_vols = [portfolio_volatility(w, cov) for w in random_weights]
        assert mvp.volatility <= min(random_vols) + 1e-6

    def test_short_selling_mvp_lower_or_equal_volatility(self, sample_markets):
        cov = sample_markets["cov"]
        mvp_long = find_min_variance_portfolio(cov, allow_short=False)
        mvp_short = find_min_variance_portfolio(cov, allow_short=True)

        assert np.isclose(np.sum(mvp_short.weights), 1.0, atol=1e-6)
        assert mvp_short.volatility <= mvp_long.volatility + 1e-6


class TestTangencyPortfolio:
    def test_long_only_tangency_beats_random_portfolios(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        rf = sample_markets["rf"]

        tangency = find_tangency_portfolio(returns, cov, rf, allow_short=False)

        assert np.isclose(np.sum(tangency.weights), 1.0, atol=1e-6)
        assert np.all(tangency.weights >= -1e-8)
        assert tangency.sharpe_ratio is not None

        rng = np.random.default_rng(123)
        random_weights = rng.dirichlet(np.ones(len(returns)), size=5000)
        random_sharpes = [
            portfolio_sharpe_ratio(w, returns, cov, rf) for w in random_weights
        ]
        assert tangency.sharpe_ratio >= max(random_sharpes) - 1e-5

    def test_short_selling_tangency_higher_or_equal_sharpe(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        rf = sample_markets["rf"]

        tang_long = find_tangency_portfolio(returns, cov, rf, allow_short=False)
        tang_short = find_tangency_portfolio(returns, cov, rf, allow_short=True)

        assert np.isclose(np.sum(tang_short.weights), 1.0, atol=1e-6)
        assert tang_short.sharpe_ratio >= tang_long.sharpe_ratio - 1e-6

    def test_rf_higher_than_all_returns_handled_gracefully(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        high_rf = float(np.max(returns)) + 0.05  # 14.5% > all expected returns

        tang_long = find_tangency_portfolio(returns, cov, high_rf, allow_short=False)
        assert np.isclose(np.sum(tang_long.weights), 1.0, atol=1e-6)
        assert np.all(tang_long.weights >= -1e-8)
        assert tang_long.sharpe_ratio is not None
        assert tang_long.sharpe_ratio < 0.0


class TestEfficientFrontier:
    def test_long_only_frontier_shape_and_monotonicity(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]

        frontier = compute_efficient_frontier(
            returns, cov, num_points=40, allow_short=False
        )
        mvp = find_min_variance_portfolio(cov, allow_short=False)

        assert len(frontier.returns) == 40
        assert len(frontier.volatilities) == 40
        assert frontier.weights.shape == (40, len(returns))

        # Every portfolio sums to 1 and respects long-only bounds
        assert np.allclose(np.sum(frontier.weights, axis=1), 1.0, atol=1e-5)
        assert np.all(frontier.weights >= -1e-8)

        # First point matches MVP volatility
        assert np.isclose(frontier.volatilities[0], mvp.volatility, atol=1e-5)

        # Upper efficient frontier has non-decreasing volatility as return rises
        assert np.all(np.diff(frontier.returns) >= -1e-6)
        assert np.all(np.diff(frontier.volatilities) >= -1e-5)

    def test_short_selling_frontier_valid(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]

        frontier = compute_efficient_frontier(
            returns, cov, num_points=30, allow_short=True
        )
        assert len(frontier.returns) == 30
        assert np.allclose(np.sum(frontier.weights, axis=1), 1.0, atol=1e-5)
        assert np.all(np.diff(frontier.volatilities) >= -1e-5)


class TestCALTargetPortfolio:
    def test_cal_by_target_return_lending(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        rf = sample_markets["rf"]
        tangency = find_tangency_portfolio(returns, cov, rf, allow_short=False)

        target_ret = 0.5 * (rf + tangency.expected_return)
        cal = find_cal_portfolio(tangency, rf, target_return=target_ret)

        assert np.isclose(cal.expected_return, target_ret, atol=1e-10)
        assert np.isclose(cal.risky_allocation, 0.5, atol=1e-10)
        assert np.isclose(cal.risk_free_weight, 0.5, atol=1e-10)
        assert np.isclose(
            np.sum(cal.risky_weights) + cal.risk_free_weight, 1.0, atol=1e-10
        )
        assert np.isclose(cal.volatility, 0.5 * tangency.volatility, atol=1e-10)
        assert np.isclose(cal.sharpe_ratio, tangency.sharpe_ratio, atol=1e-10)

    def test_cal_by_target_return_leveraged(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        rf = sample_markets["rf"]
        tangency = find_tangency_portfolio(returns, cov, rf, allow_short=False)

        target_ret = rf + 1.5 * (tangency.expected_return - rf)
        cal = find_cal_portfolio(tangency, rf, target_return=target_ret)

        assert np.isclose(cal.expected_return, target_ret, atol=1e-10)
        assert np.isclose(cal.risky_allocation, 1.5, atol=1e-10)
        assert np.isclose(cal.risk_free_weight, -0.5, atol=1e-10)
        assert np.isclose(
            np.sum(cal.risky_weights) + cal.risk_free_weight, 1.0, atol=1e-10
        )
        assert np.isclose(cal.sharpe_ratio, tangency.sharpe_ratio, atol=1e-10)

    def test_cal_by_target_volatility_and_round_trip(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        rf = sample_markets["rf"]
        tangency = find_tangency_portfolio(returns, cov, rf, allow_short=False)

        target_vol = 0.12
        cal_vol = find_cal_portfolio(tangency, rf, target_volatility=target_vol)

        assert np.isclose(cal_vol.volatility, target_vol, atol=1e-10)
        assert np.isclose(
            np.sum(cal_vol.risky_weights) + cal_vol.risk_free_weight, 1.0, atol=1e-10
        )
        assert np.isclose(cal_vol.sharpe_ratio, tangency.sharpe_ratio, atol=1e-10)

        # Round-trip: passing the resulting return back gives the same portfolio
        cal_ret = find_cal_portfolio(
            tangency, rf, target_return=cal_vol.expected_return
        )
        assert np.isclose(cal_ret.volatility, cal_vol.volatility, atol=1e-10)
        assert np.allclose(cal_ret.risky_weights, cal_vol.risky_weights, atol=1e-10)

    def test_cal_zero_volatility_is_all_risk_free(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        rf = sample_markets["rf"]
        tangency = find_tangency_portfolio(returns, cov, rf, allow_short=False)

        cal = find_cal_portfolio(tangency, rf, target_volatility=0.0)
        assert np.isclose(cal.risky_allocation, 0.0, atol=1e-12)
        assert np.isclose(cal.risk_free_weight, 1.0, atol=1e-12)
        assert np.isclose(cal.expected_return, rf, atol=1e-12)
        assert np.allclose(cal.risky_weights, 0.0, atol=1e-12)

    def test_cal_rejects_invalid_inputs(self, sample_markets):
        returns = sample_markets["returns"]
        cov = sample_markets["cov"]
        rf = sample_markets["rf"]
        tangency = find_tangency_portfolio(returns, cov, rf, allow_short=False)

        with pytest.raises(ValueError, match="exactly one"):
            find_cal_portfolio(tangency, rf)
        with pytest.raises(ValueError, match="exactly one"):
            find_cal_portfolio(tangency, rf, target_return=0.08, target_volatility=0.12)
        with pytest.raises(ValueError, match="non-negative"):
            find_cal_portfolio(tangency, rf, target_volatility=-0.05)

        high_rf = float(np.max(returns)) + 0.05
        bad_tangency = find_tangency_portfolio(returns, cov, high_rf, allow_short=False)
        with pytest.raises(ValueError, match="positive-slope CAL"):
            find_cal_portfolio(bad_tangency, high_rf, target_return=0.10)
