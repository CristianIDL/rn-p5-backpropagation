'''
xor.py: Implementa la red neuronal para resolver el problema XOR utilizando la clase RedNeuronal.
'''

from red_neuronal import *

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

red = RedNeuronal(2, 2, 1, lr=0.5)
errores = red.train(X, y, epochs=5000, graficar=True)

print("Predicciones:")
print(red.predict(X))