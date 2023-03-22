import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import Ridge


class RBF_network:
    """
    metric='sqeuclidean'とすることで
    ユークリッド距離の二乗を計算
    """
    def __init__(self, centers, beta, alpha=0):
        self.kernel = np.exp
        self.centers = centers
        self.beta = beta
        self.reg = Ridge(alpha=alpha, fit_intercept=False)

    def fit(self, X, Y, return_K=False):
        dist_mat = distance.cdist(X, self.centers, metric='sqeuclidean')
        # K = self.kernel(-self.beta*dist_mat)
        K = np.hstack([np.ones((len(X), 1)), X, self.kernel(-self.beta*dist_mat)])
        self.reg.fit(K, Y)
        if return_K:
            return self.reg.coef_, K
        else:
            return self.reg.coef_

    def predict(self, x, Wout_opt):
        dist_mat = distance.cdist(x, self.centers, metric='sqeuclidean')
        # K = self.kernel(-self.beta*dist_mat)
        K = np.hstack([np.ones((len(x), 1)), x, self.kernel(-self.beta*dist_mat)])
        return np.dot(Wout_opt, K.T).T