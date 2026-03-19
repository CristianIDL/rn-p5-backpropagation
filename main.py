'''
main.py: Implementa el algoritmo del perceptrón.
'''

from perceptron import perceptron, imprimir_actual
import numpy as np

def main():
    # Establecemos las entradas de una compuerta
    x = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]]) 

    # Establecemos los targets de una compuerta AND y OR
    t_AND = np.array([[0],[0],[0],[1]])
    t_OR = np.array([[0],[1],[1],[1]])
    
    # Necesario para el XOR
    t_XOR1 = np.array([[0],[1],[1],[1]])
    t_XOR2 = np.array([[1],[1],[1],[0]])


    # Establecemos los pesos
    w = np.array([0.7, 
                  0.7])
    b = 0.5
    a = 0.5

    convergencia = False

    for i in range(1, 7):
        print(f"\n ===  Época {i} ===")
        w, b, a, convergencia = perceptron(x,w,b,t_AND,convergencia,a,epoca=i)
        print(f"Pesos y bias después de la época {i}:")
        imprimir_actual(w,b,a)
        if(convergencia):
            print(f"\nConvergencia alcanzadaa al final de la época {i}")
            break

if __name__ == "__main__":
    main()