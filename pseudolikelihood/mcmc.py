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
    n_sites, n_features = features.shape
    n_neighbors = A.sum(axis=1).reshape(-1, 1)

    betas = model.coef_[:, :n_features]
    eta = model.coef_[:, n_features:]

    p = safe_sparse_dot(features, betas.T, dense_output=True)
    p += model.intercept_

    p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
    p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
    p_nonspatial = np.exp(p_nonspatial, p_nonspatial)

    i = 0

    sample = multinomial(n_observations, p_nonspatial)
    for i in range(1000):
        print(i)
        spatial = safe_sparse_dot(A, (sample/n_observations - p_nonspatial))/n_neighbors
        spatial[np.isnan(spatial)] = 0

        sample_p = p.copy()
        for k in range(n_classes - 1):
            sample_p[:, k:k+1] += safe_sparse_dot(np.delete(spatial, k, axis = 1),
                                                  eta[k:k+1, :].T)

        sample_p = softmax(np.hstack((sample_p, np.zeros((features.shape[0], 1)))))

        sample = multinomial(n_observations, sample_p)

    return sample
