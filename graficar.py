'''
graficar.py: Contiene funciones para graficar las fronteras de decisión.
'''

import numpy as np
import matplotlib.pyplot as plt

def graficar_red(X, y, red, epoca):
    # Crear malla
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predicciones sobre la malla
    Z = red.predict(grid)
    Z = Z.reshape(xx.shape)

    # Graficar frontera
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # Graficar puntos reales
    class_0 = X[y.flatten() == 0]
    class_1 = X[y.flatten() == 1]

    plt.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Clase 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Clase 1')

    plt.title(f"Época {epoca}")
    plt.legend()
    plt.grid(True)

    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

def graficar_iris(X, y, redes, epoca, guardar=False, nombre_archivo="iris.png"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predecir con todas las redes
    salidas = []
    for red in redes:
        salida = red.forward(grid)
        salidas.append(salida)

    salidas = np.hstack(salidas)
    Z = np.argmax(salidas, axis=1)
    Z = Z.reshape(xx.shape)

    # Colorear regiones
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Dibujar puntos reales
    for clase in np.unique(y):
        puntos = X[y == clase]
        plt.scatter(puntos[:, 0], puntos[:, 1], label=f"Clase {clase}")

    plt.title(f"Frontera de decisión - Época {epoca}")
    plt.legend()
    plt.grid(True)

    if guardar:
        plt.savefig(nombre_archivo, dpi=300)
        print(f"Gráfica guardada como {nombre_archivo}")

    plt.show(block=False)
    plt.pause(0.5)
    plt.close()