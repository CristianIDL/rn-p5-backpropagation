'''
perceptron.py: Implementa el algoritmo del perceptrón con función escalón y reajuste de pesos.
'''

from graficar import visualizar_iteracion
import numpy as np

# Función de activación escalón
def escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

def imprimir_actual(w,b,a):
    print("w =", w)
    print("b =", b)
    print("a =", a)

def perceptron(x,w,b,t,convergencia,a=0.5,epoca=0):
    # Definimos el número de iteraciones para la época actual
    iteracion = 0
    convergencia = True  # Asumimos convergencia hasta que se demuestre lo contrario

    imprimir_actual(w,b,a)
    
    # Calculamos la salida del perceptrón para cada entrada
    for i in range(len(x)):
        # Incrementamos el contador de iteraciones
        iteracion += 1

        # Calculamos la suma ponderada de las entradas
        z = np.dot(x[i],w) + b
        print(f"\tz[{i}] = {x[i]}*{w} + {b} = {z:.2f}")

        # Aplicamos la función de activación escalón
        z_i = escalon(z)
        print(f"\tescalón(z[{i}]) = {z_i}")

        if z_i != t[i][0]:
            convergencia = False
            # Calculamos el error
            print(f"\t{z_i} != {t[i][0]}\tSalida incorrecta para x[{i}]")      
            e = t[i][0] - z_i 
            # Reajustamos los pesos y el bias
            w = reajustar_pesos(w, a, e, x[i])
            b = reajustar_bias(b, a, e)
            print(f"Reajustamos pesos:")
            imprimir_actual(w,b,a)
            visualizar_iteracion(x,w,b,t,epoca,iteracion,f"Iteración {iteracion}")

    return w,b,a,convergencia

# Numpy tiene la ventaja del broadcasting, 
# Permitiendo realizar operaciones entre arrays y escalare rápidamente.

def reajustar_pesos(w, a, e,x):
    w = w + (a * e * x)
    return w

def reajustar_bias(b, a, e):
    b = b + (a * e)
    return b