import numpy as np


def grid_search(diff_i, i, perplexity):
    result = np.inf

    norm = np.linalg.norm(diff_i, axis=1)

    for sigma_search in np.linspace(0.01 * np.std(norm), 5 * np.std(norm), 200):
        p = np.exp(-norm ** 2 / (2 * sigma_search ** 2))
        p[i] = 0

        p_new = np.maximum(p / np.sum(p), 10 ** (-4))

        H = -np.sum(p_new * np.log2(p_new))

        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            sigma = sigma_search

    return sigma


def get_original_pairwise_affinities(X, perplexity=10):
    n = len(X)
    p_ij = np.zeros(shape=(n, n))

    for i in range(0, n):
        sigma_i = grid_search(X[i] - X, i, perplexity)
        norm = np.linalg.norm(X[i] - X, axis=1)
        p_ij[i, :] = np.exp(-norm ** 2 / (2 * sigma_i ** 2))

        np.fill_diagonal(p_ij, 0)
        p_ij[i, :] = p_ij[i, :] / np.sum(p_ij[i, :])

    p_ij = np.maximum(p_ij, 10 ** (-4))

    return p_ij


def get_symmetric_p_ij(p_ij: np.array([])):
    n = len(p_ij)
    p_ij_symmetric = np.zeros(shape=(n, n))

    for i in range(0, n):
        for j in range(0, n):
            p_ij_symmetric[i, j] = (p_ij[i, j] + p_ij[j, i]) / (2 * n)

    p_ij_symmetric = np.maximum(p_ij_symmetric, (10 ** (-4)))

    return p_ij_symmetric
