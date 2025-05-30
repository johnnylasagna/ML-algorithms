import numpy as np

class LogisticRegression:

    def __init__(self, lr, n_iter, rp):
        self.lr = lr
        self.n_iter = n_iter
        self.rp = rp
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.costs = []

        for i in range(self.n_iter):
            y_predict = np.array([self.predict(x) for x in X])
            error = y_predict - y
            _w = self.w - self.lr * (np.dot(error, X) / X.shape[0] + self.rp * self.w / X.shape[0])
            _b = self.b - self.lr * np.mean(error)
            self.w = _w
            self.b = _b
            self.costs.append(self.cost(y, y_predict))
            
    def cost(self, y, y_predict):
        return -1 * np.sum(np.dot(y, np.log(y_predict)) + np.dot(1-y, np.log(1-y_predict))) / y.shape[0] + 1 / (2 * y.shape[0]) * self.rp * np.dot(self.w, self.w)

    def predict(self, x):
        z = np.dot(self.w,x) + self.b
        fz = 1 / (1 + np.exp(-z))
        return fz