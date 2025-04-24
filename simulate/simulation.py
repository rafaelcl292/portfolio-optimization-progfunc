import numpy as np


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
    # Amostrar de Dirichlet
    raw = rng.dirichlet(np.ones(n_assets), size=n_samples)
    # Projetar cada vetor no simplex com bound por coordenada
    def _proj(vec: np.ndarray) -> np.ndarray:
        # projecão Euclidiana em {w >= 0, w <= max_weight, sum(w)=1}
        x = vec.copy()
        # bisection para achar t tal que sum(clamp(x - t, 0, max_weight)) = 1
        low, high = np.min(x - max_weight), np.max(x)
        for _ in range(50):
            t = (low + high) / 2
            w = np.minimum(np.maximum(x - t, 0), max_weight)
            s = w.sum()
            if s > 1:
                low = t
            else:
                high = t
        return np.minimum(np.maximum(x - high, 0), max_weight)
    W = np.vstack([_proj(raw[i]) for i in range(n_samples)])

    # Retorno e risco da carteira
    # Retorno esperado e risco da carteira
    mu_p = W.dot(mu)
    # variância: w' Sigma w
    var_p = np.einsum("ij,ij->i", W.dot(sigma), W)
    sigma_p = np.sqrt(var_p)

    # Calcular Sharpe Ratios
    # evitar divisão por zero
    sr = np.where(sigma_p > 0, (mu_p - r_free) / sigma_p, -np.inf)

    # Selecionar melhor amostra
    idx_best = int(np.argmax(sr))
    return float(sr[idx_best]), W[idx_best]
