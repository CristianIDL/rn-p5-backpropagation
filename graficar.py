'''
graficar.py: Contiene funciones para graficar las fronteras de decisión.
'''

import matplotlib.pyplot as plt
import numpy as np
    
def frontera_decision(ax, w, b, x_range=[-0.5, 1.5]):
    '''Dibuja la frontera de decisión del perceptrón para visualizarla en el gráfico.'''

    x1 = np.linspace(x_range[0], x_range[1], 100)  # Generamos una línea hecha de 100 puntos

    if w[1] != 0:  # Evitamos división por cero
        x2 = -(w[0] * x1 + b) / w[1]  # Calculamos x2 usando la ecuación de la recta
        ax.plot(x1, x2, 'g-', linewidth=2, label='Frontera de decisión')


def visualizar_iteracion(x,w,b,t,epoca,iteracion,titulo=""):
    '''Visualiza la iteración actual del perceptrón.'''
    fig, ax = plt.subplots(figsize=(6,6))

    # Dibujamos los ejes y la frontera de decisión
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)

    # Graficamos las entradas
    class_0 = x[t.flatten() == 0]
    class_1 = x[t.flatten() == 1]

    ax.scatter(class_0[:, 0], class_0[:, 1], 
               color='red', marker='s', label='Clase 0')
    
    ax.scatter(class_1[:, 0], class_1[:, 1], 
               color='blue', marker='o', label='Clase 1')

    # Dibujamos la frontera de decisión
    frontera_decision(ax, w, b, x_range=[0, 8])

    # Configuramos el espacio de decisión
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    # Aplicamos estilo al gráfico
    ax.grid(True, alpha=0.3)

    ax.set_title(f'Época {epoca}, Iteración {iteracion} \n w=[{w[0]:.2f}, {w[1]:.2f}], b={b:.2f}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()