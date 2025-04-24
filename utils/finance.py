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
