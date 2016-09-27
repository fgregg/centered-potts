import numpy as np

from scipy import optimize

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import logsumexp, safe_sparse_dot, squared_norm



class CenteredPotts(object):
    def __init__(self, tol=1e-4, max_iter=100, verbose=0, C=1.0):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
" class: %r" % classes_[0])

        sample_weight = np.ones(X.shape[0])

        lbin = LabelBinarizer()
        Y_multi = lbin.fit_transform(y)
        if Y_multi.shape[1] == 1:
            Y_multi = np.hstack([1 - Y_multi, Y_multi])
            
        w0 = np.zeros((self.classes_.size, n_features + 1),
                      order='F')

        w0 = w0.ravel()
        target = Y_multi

        func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]

        w0, loss, info = optimize.fmin_l_bfgs_b(
            func, w0, fprime=None,
            args=(X, target, 1. / self.C, sample_weight),
            iprint=(self.verbose > 0) - 1, pgtol=self.tol, maxiter=self.max_iter)

        if info["warnflag"] == 1 and self.verbose > 0:
            warnings.warn("lbfgs failed to converge. Increase the number "
                          "of iterations.")

        n_iter_i = info['nit'] - 1

        self.coef_ = np.reshape(w0, (self.classes_.size, -1))

        self.intercept_ = self.coef_[:, -1]
        self.coef_ = self.coef_[:, :-1]

        return self
        
        
    def _loss(self, X, y):
        pass

    def _loss_grad(self, X, y):
        pass

    def predict_proba(self, X):
        pass




def _multinomial_loss(w, X, Y, alpha, sample_weight):
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
    n_features = X.shape[1]
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    intercept = w[:, -1]
    w = w[:, :-1]
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w


def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.
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
    n_features = X.shape[1]
    grad = np.zeros((n_classes, n_features + 1))
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p

import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


iris = load_iris()
target = iris.target_names[iris.target]
print(target)


lr = LogisticRegression(fit_intercept=True,
                        multi_class='multinomial',
                        solver='lbfgs')
lr.fit(iris.data, iris.target)

print(lr.coef_)


lr = CenteredPotts()
lr.fit(iris.data, iris.target)

print(lr.coef_)


