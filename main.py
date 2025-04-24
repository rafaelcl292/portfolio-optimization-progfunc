import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from data_loader.data_loader import load_price_data
from simulate.simulation import simulate_subset
from utils.finance import annualize_covariance, annualize_returns, compute_daily_returns


def process_combination(args):
    subset, mu_all, sigma_all, n_samples, r_free = args
    idx = list(subset)
    mu_sub = mu_all[idx]
    sigma_sub = sigma_all[np.ix_(idx, idx)]
    sr, w = simulate_subset(mu_sub, sigma_sub, len(idx), n_samples, r_free)
    return subset, sr, w


def main():
    import argparse

    # Argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Otimização de Carteiras Dow Jones")
    parser.add_argument(
        "--select", type=int, default=25, help="Número de ativos na carteira"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Número de simulações por combinação"
    )
    parser.add_argument(
        "--free-rate", type=float, default=0.0, help="Taxa livre de risco anual"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Número de processos paralelos"
    )
    args = parser.parse_args()

    # Configurações
    tickers = [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CRM",
        "CSCO",
        "CVX",
        "DIS",
        "DOW",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PG",
        "TRV",
        "UNH",
        "V",
        "VZ",
        "WBA",
        "WMT",
    ]
    start, end = "2024-08-01", "2024-12-31"
    n_assets = len(tickers)
    n_select = args.select
    n_samples = args.samples
    r_free = args.free_rate
    n_workers = args.workers or (os.cpu_count() or 1)

    # Carregar dados
    price_df = load_price_data(tickers, start, end)
    daily_returns = compute_daily_returns(price_df)
    mu = annualize_returns(daily_returns).to_numpy()
    sigma = annualize_covariance(daily_returns).to_numpy()

    # Gerar combinações
    combos = itertools.combinations(range(n_assets), n_select)
    args_iter = ((subset, mu, sigma, n_samples, r_free) for subset in combos)

    best_sr = -np.inf
    best_res = None
    # Paralelização

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for subset, sr, w in executor.map(process_combination, args_iter, chunksize=10):
            if sr > best_sr:
                best_sr = sr
                best_res = (subset, sr, w)

    # Exibir resultado
    subset_idx, sr, w = best_res
    best_tickers = [tickers[i] for i in subset_idx]
    print(f"Melhor Sharpe Ratio: {sr:.4f}")
    print("Tickers:", best_tickers)
    print("Weights:", w)


if __name__ == "__main__":
    main()
