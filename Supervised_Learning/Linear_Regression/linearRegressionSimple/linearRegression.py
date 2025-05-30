import numpy as np

class LinearRegression:

    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter  
    
    def fit(self, X, y):
        self.w = 0
        self.b = 0

        for i in range(self.n_iter):
            output = np.array([self.predict(x) for x in X])
            error = output - y
            _w = self.w - self.lr * np.dot(error, X) / X.shape[0]
            _b = self.b - self.lr * sum(error) / X.shape[0]
            self.w = _w
            self.b = _b
            # print(self.w, self.b)

    def predict(self, x):
        return self.w * x + self.b
            
