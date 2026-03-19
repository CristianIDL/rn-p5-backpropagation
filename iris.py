'''
iris.py: Implementa el algoritmo del perceptrón para clasificar el conjunto de datos Iris.
'''

from perceptron import perceptron, imprimir_actual
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def cargar_datos():
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(iris.feature_names)

    # Cargamos los datos del conjunto de datos Iris
    X = X[:, [2, 3]]  # Seleccionamos solo las columnas de sepal_length y petal_length

    return X, y

def main():
    # Cargamos los datos del conjunto de datos Iris
    X, y = cargar_datos()

    # Seleccionamos la sepal_lenght y petal_length para la clasificación
    print(X)  # Debería mostrar (150, 2)
    print(y)  # 0 = setosa, 1 = versicolor, 2 = virginica

    # Convertimos las etiquetas a formato binario (0 para setosa, 1 para versicolor y virginica)
    t_setosa = (y == 0).astype(int).reshape(-1, 1)  # Clase 0: setosa
    t_versicolor = (y == 1).astype(int).reshape(-1, 1)  # Clase 1: versicolor
    t_virginica = (y == 2).astype(int).reshape(-1, 1)  # Clase 2: virginica

    print("Etiquetas para la clase Setosa:\n", t_setosa)
    print("Etiquetas para la clase Versicolor:\n", t_versicolor)
    print("Etiquetas para la clase Virginica:\n", t_virginica)

    w = np.array([0.7, 
                  0.7])
    b = 0.5
    a = 0.5

    convergencia = False

    for i in range(1, 20):
        print(f"\n ===  Época {i} ===")
        w, b, a, convergencia = perceptron(X, w, b, t_versicolor, convergencia, a, epoca=i)
        print(f"Pesos y bias después de la época {i}:")
        imprimir_actual(w,b,a) 
        if(convergencia):
            print(f"\nConvergencia alcanzadaa al final de la época {i}")
            break

    
if __name__ == "__main__":
    main()