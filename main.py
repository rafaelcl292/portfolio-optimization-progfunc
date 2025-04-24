import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from data_loader.data_loader import load_price_data
from simulate.simulation import simulate_subset
from utils.finance import annualize_covariance, annualize_returns, compute_daily_returns


def process_combination(args):
    # args: (seed_offset, subset, mu_all, sigma_all, n_samples, r_free)
    seed_offset, subset, mu_all, sigma_all, n_samples, r_free = args
    idx = list(subset)
    mu_sub = mu_all[idx]
    sigma_sub = sigma_all[np.ix_(idx, idx)]
    # derive seed per combination
    seed = seed_offset
    sr, w = simulate_subset(mu_sub, sigma_sub, len(idx), n_samples, r_free, seed=seed)
    return subset, sr, w


def optimize_portfolio(mu, sigma, tickers, n_select, n_samples, r_free, workers, base_seed):
    """
    Retorna (best_sr, best_subset, best_weights) otimizado.
    """
    import itertools
    from concurrent.futures import ProcessPoolExecutor

    n_assets = len(tickers)
    combos = itertools.combinations(range(n_assets), n_select)
    # Enumerate for reproducible seeds
    args_iter = (
        (base_seed + i, subset, mu, sigma, n_samples, r_free)
        for i, subset in enumerate(combos)
    )
    best_sr = -np.inf
    best_res = None
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for subset, sr, w in executor.map(process_combination, args_iter, chunksize=10):
            if sr > best_sr:
                best_sr = sr
                best_res = (subset, w)
    subset_idx, weights = best_res
    return best_sr, subset_idx, weights

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
    parser.add_argument(
        "--seed", type=int, default=0, help="Semente base para RNG"
    )
    parser.add_argument(
        "--benchmark", type=int, default=0, help="Número de execuções para comparar tempo (serial vs paralelo)"
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
    base_seed = args.seed
    n_bench = args.benchmark

    # Carregar dados
    price_df = load_price_data(tickers, start, end)
    daily_returns = compute_daily_returns(price_df)
    mu = annualize_returns(daily_returns).to_numpy()
    sigma = annualize_covariance(daily_returns).to_numpy()

    # Se benchmark ativado, comparar serial x paralelo
    if n_bench > 0:
        import time
        times = {"serial": [], "parallel": []}
        for mode in [("serial", 1), ("parallel", n_workers)]:
            label, workers = mode
            for _ in range(n_bench):
                t0 = time.perf_counter()
                optimize_portfolio(mu, sigma, tickers, n_select, n_samples, r_free, workers, base_seed)
                times[label].append(time.perf_counter() - t0)
        print("Benchmark resultados (segundos):")
        for label in ["serial", "parallel"]:
            arr = times[label]
            print(f" {label}: mean={np.mean(arr):.2f}, std={np.std(arr):.2f}")
        return

    # Execução padrão
    best_sr, subset_idx, w = optimize_portfolio(
        mu, sigma, tickers, n_select, n_samples, r_free, n_workers, base_seed
    )
    best_tickers = [tickers[i] for i in subset_idx]
    print(f"Melhor Sharpe Ratio: {best_sr:.4f}")
    print("Tickers:", best_tickers)
    print("Weights:", w)


if __name__ == "__main__":
    main()
