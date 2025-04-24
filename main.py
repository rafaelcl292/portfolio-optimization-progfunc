import argparse
import itertools
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from data_loader.data_loader import load_price_data
from utils.finance import (
    annualize_covariance,
    annualize_returns,
    compute_daily_returns,
    project_weights,
    best_sharpe,
)

"""
Worker initializer: set global data for combination processing.
"""
GLOBAL_W: np.ndarray
GLOBAL_MU: np.ndarray
GLOBAL_SIGMA: np.ndarray
GLOBAL_RF: float
GLOBAL_N: int


def init_worker(Wsamples, mu_arr, sigma_arr, r_free, n_select):
    global GLOBAL_W, GLOBAL_MU, GLOBAL_SIGMA, GLOBAL_RF, GLOBAL_N
    GLOBAL_W = Wsamples
    GLOBAL_MU = mu_arr
    GLOBAL_SIGMA = sigma_arr
    GLOBAL_RF = r_free
    GLOBAL_N = n_select


def process_combination(subset):
    # subset: tuple of asset indices
    idx = list(subset)
    mu_sub = GLOBAL_MU[idx]
    sigma_sub = GLOBAL_SIGMA[np.ix_(idx, idx)]
    W = GLOBAL_W  # shape (n_samples, n_select)
    sr_best, w_best = best_sharpe(mu_sub, sigma_sub, W, GLOBAL_RF)
    return subset, sr_best, w_best


def optimize_portfolio(
    mu, sigma, tickers, n_select, n_samples, r_free, workers, base_seed
):
    """
    Retorna (best_sr, best_subset, best_weights) otimizado.
    """

    n_assets = len(tickers)
    # Prepare all combinations of asset indices
    combos = itertools.combinations(range(n_assets), n_select)
    total = math.comb(n_assets, n_select)
    # Pre-generate sample weights once for all combinations
    rng = np.random.default_rng(base_seed)
    raw = rng.dirichlet(np.ones(n_select), size=n_samples)
    # Project samples onto bounded simplex
    Wsamples = np.vstack(
        [project_weights(raw[i], max_weight=0.2) for i in range(n_samples)]
    )
    best_sr = -np.inf
    best_res = None
    # Determine chunksize to balance overhead and responsiveness
    chunksize = max(1, total // (workers * 20))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(Wsamples, mu, sigma, r_free, n_select),
    ) as executor:
        task_iter = executor.map(process_combination, combos, chunksize=chunksize)
        for subset, sr, w in tqdm(
            task_iter,
            total=total,
            desc="Processando combinações",
            unit="combo",
            miniters=chunksize,
        ):
            if sr > best_sr:
                best_sr = sr
                best_res = (subset, w)
    subset_idx, weights = best_res
    return best_sr, subset_idx, weights


def main():
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
    parser.add_argument("--seed", type=int, default=0, help="Semente base para RNG")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Número de execuções para comparar tempo (serial vs paralelo)",
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
    print(f"Carregando dados para {len(tickers)} tickers de {start} a {end}...")
    price_df = load_price_data(tickers, start, end)
    daily_returns = compute_daily_returns(price_df)
    mu = annualize_returns(daily_returns).to_numpy()
    sigma = annualize_covariance(daily_returns).to_numpy()

    # Se benchmark ativado, comparar serial x paralelo
    if n_bench > 0:
        times = {"serial": [], "parallel": []}
        for mode in [("serial", 1), ("parallel", n_workers)]:
            label, workers = mode
            for _ in range(n_bench):
                t0 = time.perf_counter()
                optimize_portfolio(
                    mu, sigma, tickers, n_select, n_samples, r_free, workers, base_seed
                )
                times[label].append(time.perf_counter() - t0)
        print("Benchmark resultados (segundos):")
        for label in ["serial", "parallel"]:
            arr = times[label]
            print(f" {label}: mean={np.mean(arr):.2f}, std={np.std(arr):.2f}")
        return

    total_combos = math.comb(n_assets, n_select)
    print(
        f"Iniciando otimização com {total_combos} combinações, {n_samples} simulações cada, r_free={r_free}"
    )
    best_sr, subset_idx, w = optimize_portfolio(
        mu, sigma, tickers, n_select, n_samples, r_free, n_workers, base_seed
    )
    best_tickers = [tickers[i] for i in subset_idx]
    print(f"Melhor Sharpe Ratio: {best_sr:.4f}")
    print("Tickers:", best_tickers)
    print("Weights:", w)


if __name__ == "__main__":
    main()
