import numpy as np

class LDA1dThres:
    def __init__(self):
        self.S_b = np.array([])
        self.S_w = np.array([])
        self.threshold = 0
        self.w = np.array([])

    def fit(self, X, y):
        m0 = np.mean(X[y == 0, :], axis = 0).reshape(X.shape[1],1)
        m1 = np.mean(X[y == 1, :], axis = 0).reshape(X.shape[1],1)
        self.S_b = (m0-m1)@(m0-m1).T
        self.S_w = np.cov(X[y == 0, :].T)
        self.S_w += np.cov(X[y == 1, :].T)
        S_w_inverse = np.linalg.inv(self.S_w)
        self.w = S_w_inverse@((m0-m1))

        m = np.mean(X, axis = 0).reshape(X.shape[1],1)
        self.threshold = self.w.T@m
        return self
    
    def predict(self, X):
        new_X = self.w.T@X.T
        y = np.zeros((1,X.shape[0]))
        y[new_X <= self.threshold] = 1
        return y.ravel()