import numpy as np

# gradient descent

class logisticRegression:
    def __init__(self, k, d, step_size =  0.01, max_iter = 2000, tol = 0.001):
        self.w = np.random.uniform(-0.01, 0.01, (d+1, k))
        self.max_iter = max_iter
        self.k = k
        self.step_size = step_size
        self.d = d
        self.last_error = 0
        self.tol = tol

    def calc_error(self, X, y):
        eps = 1e-15
        new_y = X@self.w
        # soft max
        for i in range(new_y.shape[0]):
            new_y[i] = new_y[i]-np.max(new_y[i])
            new_y[i] = np.exp(new_y[i])
            sum_y = sum(new_y[i])
            new_y[i] = new_y[i]/sum_y

        new_y = np.clip(new_y, eps, 1 - eps)
        error = 0
        # -(yt log(yp) + (1 - yt) log(1 - yp))
        new_y = np.where(new_y == 0, 1e-9, new_y)
        error = -np.sum(y*np.log(new_y))/y.shape[0]

        # error = -1*error/y.shape[0]
        return new_y, error


    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        cur_y, self.last_error = self.calc_error(X, y)

        for it in range(self.max_iter):
            delta_w = np.zeros((self.d+1, self.k))
            for i in range(self.k):
                for j in range(X.shape[0]):
                    delta_w[:,i] += (cur_y[j, i] - y[j,i])*X[j]
            
            self.w = self.w - self.step_size*delta_w
            cur_y, cur_error = self.calc_error(X, y)

            # converge
            if (cur_error < self.tol):
                return self
            
            self.last_error = cur_error
        
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        new_y = X@self.w
        # soft max
        for i in range(new_y.shape[0]):
            new_y[i] = new_y[i]-np.max(new_y[i])
            new_y[i] = np.exp(new_y[i])
            sum_y = sum(new_y[i])
            new_y[i] = new_y[i]/sum_y

        return new_y