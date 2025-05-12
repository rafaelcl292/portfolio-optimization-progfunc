import numpy as np
import pandas as pd


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos diários (p_t / p_{t-1} - 1) de um DataFrame de preços.
    """
    returns = prices.pct_change().dropna(how="all")
    return returns


def annualize_returns(
    daily_returns: pd.DataFrame, periods_per_year: int = 252
) -> pd.Series:
    """
    Calcula o retorno anualizado como média diária * períodos por ano.
    """
    mean_daily = daily_returns.mean()
    return mean_daily * periods_per_year


def annualize_covariance(
    daily_returns: pd.DataFrame, periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calcula matriz de covariância anualizada a partir dos retornos diários.
    """
    cov_daily = daily_returns.cov()
    return cov_daily * periods_per_year


def project_weights(
    x: np.ndarray, max_weight: float = 0.2, tol: int = 50
) -> np.ndarray:
    """
    Project vector x onto the simplex with box constraints:
    0 <= w_i <= max_weight, sum(w) = 1.
    Uses bisection search to find the clipping threshold.
    """
    low, high = np.min(x - max_weight), np.max(x)
    for _ in range(tol):
        t = (low + high) / 2
        w = np.minimum(np.maximum(x - t, 0), max_weight)
        if w.sum() > 1:
            low = t
        else:
            high = t
    return np.minimum(np.maximum(x - high, 0), max_weight)


def best_sharpe(
    mu: np.ndarray, sigma: np.ndarray, W: np.ndarray, r_free: float = 0.0
) -> tuple[float, np.ndarray]:
    """
    Given expected returns mu, covariance matrix sigma, and an array of
    weight vectors W (shape [n_samples, n_assets]), compute per-sample
    Sharpe ratios and return the best (sr, weights).
    """
    mu_p = W.dot(mu)
    var_p = np.einsum("ij,ij->i", W.dot(sigma), W)
    sigma_p = np.sqrt(var_p)
    # avoid division by zero
    sr = np.where(sigma_p > 0, (mu_p - r_free) / sigma_p, -np.inf)
    idx = int(np.argmax(sr))
    return float(sr[idx]), W[idx]
