import numpy as np

# LDA

class naiveBayesGaussian:
    def __init__(self, k):
        self.pi = np.array([])
        self.mu = np.array([])
        self.sigma = np.array([])
        self.w_k = np.array([])
        self.k = k

    def fit(self, X, y):
        self.sigma_all = np.zeros((X.shape[1], X.shape[1]))
        self.mu = np.zeros((self.k, X.shape[1], 1))
        self.pi = np.zeros((self.k, 1))
        self.sigma = np.zeros((self.k, X.shape[1], X.shape[1]))
        self.w0 = np.zeros((self.k, 1))
        self.w_k = np.zeros((self.k, X.shape[1], 1))
        for i in range(self.k):
            self.mu[i] = np.mean(X[y == i, :], axis = 0).reshape(X.shape[1], 1)
            self.sigma[i] = (X[y==i, :].T-self.mu[i])@(X[y==i, :].T-self.mu[i]).T
            if X[y==i, :].shape[0] > 0:
                self.sigma[i] = self.sigma[i]/(X[y==i, :].shape[0])
            # Naive-Bayes with marginal Gaussian distributions
            self.sigma[i] = np.diag(np.diag(self.sigma[i])) 
            self.pi[i] = (y[y==i].shape[0]/y.shape[0])
            self.sigma_all += self.sigma[i]*self.pi[i]

        for i in range(self.k):
            self.w_k[i] = np.linalg.pinv(self.sigma_all)@(self.mu[i])
            self.w0[i] = -1/2*(self.mu[i].T@np.linalg.pinv(self.sigma_all)@self.mu[i])
            if self.pi[i] != 0:
                self.w0[i] += np.log(self.pi[i])
        

        return self

    def predict(self, X):
        y = np.zeros((X.shape[0], 1))
        for row in range(X.shape[0]):
            temp_y = np.zeros(self.k)
            for i in range(self.k):
                temp_y[i] = np.add((X[row].reshape(1,X.shape[1]))@(self.w_k[i].reshape(X.shape[1],1)), self.w0[i])
            
            temp_y = np.exp(temp_y-np.max(temp_y))
            sum_y = sum(temp_y)
            temp_y = temp_y/sum_y
            y[row] = np.argmax(temp_y)

        return y.ravel()