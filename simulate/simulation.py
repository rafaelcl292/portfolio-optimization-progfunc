import numpy as np


def simulate_subset(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_assets: int,
    n_samples: int,
    r_free: float,
    max_weight: float = 0.2,
) -> tuple[float, np.ndarray]:
    """
    Simula n_samples de carteiras long-only com restrição de peso máximo,
    retorna o maior Sharpe Ratio e os pesos correspondentes.
    """
    # Gerar pesos via Dirichlet e projetar para respeitar peso máximo
    # Amostrar n_samples e ajustar cada vetor
    raw = np.random.dirichlet(np.ones(n_assets), size=n_samples)
    # Clipping de pesos e renormalização
    W = np.minimum(raw, max_weight)
    sums = W.sum(axis=1, keepdims=True)
    # Evitar divisão por zero (caso raro): se soma == 0, redistribuir uniformemente
    zero_mask = sums.squeeze() == 0
    if np.any(zero_mask):
        W[zero_mask] = np.ones(n_assets) / n_assets
        sums = W.sum(axis=1, keepdims=True)
    W = W / sums

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
