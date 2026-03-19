'''
red_neuronal.py: Implementa una red neuronal simple con una capa oculta para clasificación binaria.
'''

import numpy as np

class RedNeuronal:
    def __init__(self, input_size, hidden_size, output_size, lr=0.5):
        self.lr = lr
        
        # Inicialización de pesos
        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.random.rand(hidden_size)
        
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.random.rand(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)

    # 🔹 Forward
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2

    # 🔹 Backpropagation
    def backward(self, X, y, output):
        error = y - output
        
        d_output = error * self.sigmoid_deriv(output)
        
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.sigmoid_deriv(self.a1)

        # Actualización de pesos
        self.W2 += self.a1.T.dot(d_output) * self.lr
        self.b2 += np.sum(d_output, axis=0) * self.lr
        
        self.W1 += X.T.dot(d_hidden) * self.lr
        self.b1 += np.sum(d_hidden, axis=0) * self.lr

    def train(self, X, y, epochs=1000):
        errores = []
        
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            error = np.mean(np.square(y - output))
            errores.append(error)
            
            if i % 100 == 0:
                print(f"Época {i}, Error: {error}")
        
        return errores

    def predict(self, X):
        output = self.forward(X)
        return np.round(output)