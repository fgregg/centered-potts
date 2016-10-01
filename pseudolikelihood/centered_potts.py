import numpy as np

from scipy import optimize
from scipy.sparse import coo_matrix

import sklearn.preprocessing
from sklearn.utils.extmath import (logsumexp, safe_sparse_dot, squared_norm,
                                   softmax)


class CenteredPotts(object):
    def __init__(self, tol=1e-4, max_iter=100, verbose=0, C=1.0):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.C = C

    def fit(self, X, y):
        features, A = X

        self.classes_ = np.unique(y)
        n_samples, n_features = features.shape

        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
" class: %r" % classes_[0])

        sample_weight = np.ones(features.shape[0])

        self.lbin = LabelBinarizer()
        Y_multi = self.lbin.fit_transform(y)

        #neighbors = self.neighbors_from_adjacency(A, y)

        w0 = np.zeros((self.classes_.size - 1, n_features + 2),
                      order='F')

        w0 = w0.ravel()
        target = Y_multi

        func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]

        w0, loss, info = optimize.fmin_l_bfgs_b(
            func, w0, fprime=None,
            args=(features, A, target, 1. / self.C, sample_weight),
            iprint=(self.verbose > 0) - 1, pgtol=self.tol, maxiter=self.max_iter)

        if info["warnflag"] == 1 and self.verbose > 0:
            warnings.warn("lbfgs failed to converge. Increase the number "
                          "of iterations.")

        n_iter_i = info['nit'] - 1

        self.coef_ = np.reshape(w0, (self.classes_.size - 1, -1))

        self.intercept_ = self.coef_[:, 0]
        self.coef_ = self.coef_[:, 1:]
        

        return self
        
        
    def predict_proba(self, X, y):
        features, A = X
        #neighbors = self.neighbors_from_adjacency(A, y)

        Y_multi = self.lbin.fit_transform(y)

        betas = self.coef_[:, :-1]
        eta = self.coef_[:, -1]

        p = safe_sparse_dot(features, betas.T, dense_output=True)
        p += self.intercept_

        p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
        p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
        p_nonspatial = np.exp(p_nonspatial, p_nonspatial)

        spatial = (A * (Y_multi - p_nonspatial))[:, :-1]

        p += (eta * spatial)

        p = np.hstack((p, np.zeros((features.shape[0], 1))))

        return softmax(p)

    def neighbors_from_adjacency(self, A, y):
        Y_multi = self.lbin.transform(y)

        neighbors = A * Y_multi

        potts_neighbors = np.zeros((neighbors.shape[1], neighbors.shape[0], 2))
        for cls in range(neighbors.shape[1]):
            potts_neighbors[cls][:, 0] = neighbors[:, cls]
            potts_neighbors[cls][:, 1] = neighbors.sum(axis=1) - neighbors[:, cls]

        return potts_neighbors
    



def _multinomial_loss(w, features, A, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    loss : float
        Multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.
    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = features.shape[1]
    w = w.reshape(n_classes - 1, -1)
    sample_weight = sample_weight[:, np.newaxis]
    intercept = w[:, 0]
    w = w[:, 1:]
    betas = w[:, :-1]
    eta = w[:, -1:]
    p = safe_sparse_dot(features, betas.T)
    p += intercept

    p_nonspatial = np.hstack((p, np.zeros((features.shape[0], 1))))
    p_nonspatial -= logsumexp(p_nonspatial, axis=1)[:, np.newaxis]
    p_nonspatial = np.exp(p_nonspatial, p_nonspatial)

    spatial = (A * (Y - p_nonspatial))[:, :-1]

    p += (eta * spatial)

    p = np.hstack((p, np.zeros((features.shape[0], 1))))
    p -= logsumexp(p, axis=1)[:, np.newaxis]

    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p_nonspatial, p, w


def _multinomial_loss_grad(w, features, A, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    features : {array-like, sparse matrix}, shape (n_samples, n_features)
               Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
    Returns
    -------
    loss : float
        Multinomial loss.
    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = features.shape[1]
    grad = np.zeros((n_classes - 1, n_features + 2))
    loss, p_nonspatial, p, w = _multinomial_loss(w, features, A, Y,
                                                 alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)[:, :n_classes-1]
    grad[:, 1:n_features + 1] = safe_sparse_dot(diff.T, features)

    spatial = (A * (Y - p_nonspatial))[:, :-1]

    grad[:, -1] = (spatial * diff).sum(axis=0)
    grad[:, 1:n_features + 2] += alpha * w
    grad[:, 0] = diff.sum(axis=0)
    return loss, grad.ravel(), p



def rpotts(X, model):
    features, A = X
    n_sites = features.shape[0]

    R = np.random.uniform(size=n_sites).reshape(-1, 1)

    lower = np.zeros(n_sites).reshape(-1, 1)
    upper = np.ones(n_sites).reshape(-1, 1)

    i = 0
    while not np.array_equal(upper, lower):
        new_r = np.random.uniform(size=R.size).reshape(R.shape)
        R = np.hstack((new_r, R))

        lower[:] = 0
        upper[:] = 1
        
        for r in R.T:
            r = r.reshape(-1, 1)
            
            upper_p = model.predict_proba(X, upper)[:, :-1]
            lower_p = model.predict_proba(X, lower)[:, :-1]

            upper = (r > upper_p).astype(int)
            lower = (r > lower_p).astype(int)

    return lower

class LabelBinarizer(sklearn.preprocessing.LabelBinarizer):
    def transform(self, *args, **kwargs):
        X = super().transform(*args, **kwargs)
        if X.shape[1] == 1:
            X = np.hstack((X, 1 - X))
        return X


def to_adjacency(edges):
    N = edges.max() + 1
    A = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                       shape=(N, N)).tocsc()
    A += A.T

    return A
        
