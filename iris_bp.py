'''
iris_bp.py: Implementa una red neuronal para clasificar el dataset Iris utilizando backpropagation.
'''

from graficar import graficar_iris
from red_neuronal import RedNeuronal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

X = X[:, [2, 3]]  # Solo usaremos las características de pétalo para visualización

# 2. Normalizar (MUY IMPORTANTE)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. One vs All
redes = []
num_clases = 3

for clase in range(num_clases):
    print(f"\nEntrenando red para clase {clase}...")

    # Crear etiquetas binarias
    y_bin = (y_train == clase).astype(int).reshape(-1, 1)

    red = RedNeuronal(input_size=2, hidden_size=5, output_size=1, lr=0.3)
    red.train(X_train, y_bin, epochs=2000)

    redes.append(red)

# 5. Predicción
def predecir(X):
    salidas = []

    for red in redes:
        salida = red.forward(X)
        salidas.append(salida)

    salidas = np.hstack(salidas)
    
    # Elegimos la clase con mayor activación
    return np.argmax(salidas, axis=1)

# 6. Evaluación
y_pred = predecir(X_test)

accuracy = np.mean(y_pred == y_test)

print("\nPredicciones:", y_pred)
print("Reales:", y_test)
print(f"\nPrecisión: {accuracy * 100:.2f}%")

# Graficar frontera final
graficar_iris(X_train, y_train, redes, epoca="Final", 
              guardar=True, nombre_archivo="frontera_iris.png")