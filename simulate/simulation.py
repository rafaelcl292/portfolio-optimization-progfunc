import numpy as np

from utils.finance import best_sharpe, project_weights


def simulate_subset(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_assets: int,
    n_samples: int,
    r_free: float,
    max_weight: float = 0.2,
    seed: int | None = None,
) -> tuple[float, np.ndarray]:
    """
    Simula n_samples de carteiras long-only com restrição de peso máximo,
    retorna o maior Sharpe Ratio e os pesos correspondentes.
    """
    # Preparar gerador de números (para reprodutibilidade)
    rng = np.random.default_rng(seed)
    # Sample from Dirichlet and project onto bounded simplex
    raw = rng.dirichlet(np.ones(n_assets), size=n_samples)
    W = np.vstack([project_weights(raw[i], max_weight) for i in range(n_samples)])
    # Compute and return best Sharpe Ratio and weights
    return best_sharpe(mu, sigma, W, r_free)
