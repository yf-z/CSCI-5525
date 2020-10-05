import numpy as np

class LDA2dGaussGM:
    def __init__(self, k, d):
        self.S_b = np.array([])
        self.S_w = np.array([])
        self.w = np.array([])
        self.new_X = np.array([])
        self.k = k
        self.mu = np.array([])
        self.sigma = np.array([])
        self.d = d

        # bivariate gauss
        self.pi = np.array([])
        self.mu = np.array([])
        self.sigma = np.array([])
        self.w_k = np.array([])
    
    def project(self, X):
        temp_w = np.linalg.pinv(self.S_w)@self.S_b
        eigenvalue, eigenvector = np.linalg.eig(temp_w)
        eig_idx = np.argsort(np.abs(eigenvalue))
        eigenvalue = eigenvalue[eig_idx]
        eigenvector = eigenvector[:, eig_idx]
        self.w = np.real(eigenvector[:, -self.d:])


    def fit(self, X, y):
        mean_vector = np.zeros((self.k, X.shape[1], 1))
        S_k = np.zeros((self.k, X.shape[1], X.shape[1]))
        self.S_w = np.zeros((X.shape[1], X.shape[1]))
        self.S_b = np.zeros((X.shape[1], X.shape[1]))
        mean = np.mean(X, axis=0).reshape(X.shape[1], 1)
        for i in range(self.k):
            mean_vector[i] = np.mean(X[y == i, :], axis=0).reshape(X.shape[1], 1)
            X_k = X[y==i, :]
            self.S_w = np.add(self.S_w, (X_k.T-mean_vector[i])@(X_k.T-mean_vector[i]).T)
            self.S_b = np.add(self.S_b, X_k.shape[0]*((mean_vector[i] - mean)@(mean_vector[i]-mean).T))

        self.project(X)
        self.projected_X = X@self.w

        # bivariate gauss
        self.mu = np.zeros((self.k, self.projected_X.shape[1], 1))
        self.pi = np.zeros((self.k, 1))
        self.sigma = np.zeros((self.k, self.projected_X.shape[1], self.projected_X.shape[1]))
        self.sigma_all = np.zeros((self.projected_X.shape[1], self.projected_X.shape[1]))
        self.w0 = np.zeros((self.k, 1))
        self.w_k = np.zeros((self.k, self.projected_X.shape[1], 1))
        for i in range(self.k):
            self.mu[i] = np.mean(self.projected_X[y == i, :], axis = 0).reshape(self.projected_X.shape[1], 1)
            self.sigma[i] = (self.projected_X[y==i, :].T-self.mu[i])@(self.projected_X[y==i, :].T-self.mu[i]).T
            self.sigma[i] = self.sigma[i]/(self.projected_X[y==i, :].shape[0])
            self.pi[i] = (y[y==i].shape[0]/y.shape[0])
            self.sigma_all += self.sigma[i]*self.pi[i]

        for i in range(self.k):
            self.w_k[i] = np.linalg.pinv(self.sigma_all)@(self.mu[i])
            self.w0[i] = -1/2*(self.mu[i].T@np.linalg.pinv(self.sigma_all)@self.mu[i])+np.log(self.pi[i])
        

        return self

    def predict(self, X):
        new_X = X@self.w
        y = np.zeros((new_X.shape[0], 1))
        for row in range(new_X.shape[0]):
            temp_y = np.zeros(self.k)
            for i in range(self.k):
                temp_y[i] = np.add((new_X[row].reshape(1,2))@(self.w_k[i].reshape(2,1)), self.w0[i])
            
            temp_y = np.exp(temp_y-np.max(temp_y))
            sum_y = sum(temp_y)
            temp_y = temp_y/sum_y
            y[row] = np.argmax(temp_y)

        return y.ravel()