import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:

    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter  
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.costs = []

        for i in range(self.n_iter):
            output = np.array([self.predict(x) for x in X])
            error = output - y
            _w = self.w - self.lr * np.dot(error, X) / X.shape[0]
            _b = self.b - self.lr * np.mean(error)
            self.w = _w
            self.b = _b
            cost = np.dot(error,error)
            self.costs.append(cost)
            # if cost<1:
            #     break
            # print(self.w, self.b)

    def predict(self, x):
        return np.dot(self.w, x) + self.b
            
