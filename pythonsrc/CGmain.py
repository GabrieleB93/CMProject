import normFunction as nf
import conjugateGradient as CG
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time

def readMatrix(type,number):
    raw = []
    with open('Matrices/Matrix'+type+'/matrix'+type+str(number)+'.txt', 'r') as f:
        for line in f:
            raw.append(line.split())
    A = np.array(raw, dtype=float)
    return A

def main():

# Singola Matrice

    # A = readMatrix('F',9)
    # f = nf.normFunction(A)
    # optimizer = CG.conjugateGradient(f)
    #
    # time_start = time.perf_counter()
    # norm1,iterations = optimizer.ConjugateGradient()
    # time_elapsed = (time.perf_counter() - time_start)
    #
    # print("Time=%.5f" % time_elapsed)
    # plt.show()
    #
    # norm = LA.norm(A, ord = 2)
    # print("Norm = %.16f \nnorm2 = %.16f\ndiff = %.30f" % (norm ** 2, norm1, np.log10(abs(norm1 - norm ** 2))))

    #Serie di Matrici, confronto

    lastNorm1 = 0
    lastIterations = 0
    lastTime = 0
    plt.figure(figsize=(17, 13))
    ax1 = plt.subplot(4, 1, 1)

    for i in range(10):

        A = readMatrix('G',i+1)
        f = nf.normFunction(A)
        optimizer = CG.conjugateGradient(f)

        time_start = time.perf_counter()
        norm1, iterations = optimizer.ConjugateGradient(i+3)
        time_elapsed = (time.perf_counter() - time_start)

        if(i!=0):

            ax1.plot([lastIterations,iterations],[lastTime,time_elapsed], 'o-')
            ax1.set_ylabel('Times')
            ax1.set_xlabel('Iterations')

        lastNorm1 = norm1
        lastIterations = iterations
        lastTime = time_elapsed

    # Per Serie e Singola

    plt.show()

if __name__ == "__main__":
    main()
