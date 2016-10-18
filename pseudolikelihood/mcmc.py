import numpy as np
from sklearn.utils.extmath import (logsumexp, safe_sparse_dot, softmax)

import functools

def multinomial(n_observations, probs):
    sample = np.empty_like(probs)
    for i, prob in enumerate(probs):
        sample[i, :] = np.random.multinomial(n_observations[i], prob)

    return sample

def rmultinomial(X, n_observations, model):
    features, A = X

    n_classes = len(model.classes_)
    n_sites = features.shape[0]
    n_neighbors = A.sum(axis=1).reshape(-1, 1)

    betas = model.coef_[:, :-1]
    eta = model.coef_[:, -1:]

    p = safe_sparse_dot(features, betas.T, dense_output=True)
    p += model.intercept_

    p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
    p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
    p_nonspatial = np.exp(p_nonspatial, p_nonspatial)

    i = 0

    sample = multinomial(n_observations, p_nonspatial)
    for _ in range(5000):
        spatial = safe_sparse_dot(A, (sample/n_observations - p_nonspatial))[:, :-1]/n_neighbors
        spatial[np.isnan(spatial)] = 0

        sample_p = p + (eta.T * np.array(spatial))
        sample_p = softmax(np.hstack((sample_p, np.zeros((features.shape[0], 1)))))

        sample = multinomial(n_observations, sample_p)

    return sample
