import numpy as np

class LinearRegression:

    def __init__(self, lr, n_iter, rp):
        self.lr = lr
        self.n_iter = n_iter  
        self.rp = rp
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.costs = []

        for i in range(self.n_iter):
            output = np.array([self.predict(x) for x in X])
            error = output - y
            _w = self.w - self.lr * (np.dot(error, X) / X.shape[0] + self.rp * self.w / X.shape[0])
            _b = self.b - self.lr * np.mean(error)
            self.w = _w
            self.b = _b
            mse = np.dot(error,error) / y.shape[0]
            reg_cost = (self.rp / (2 * X.shape[0])) * np.sum(np.square(self.w))
            cost = mse + reg_cost
            self.costs.append(cost)

    def predict(self, x):
        return np.dot(self.w, x) + self.b

