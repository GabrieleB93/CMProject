import normFunction as nf
import steepestGradientDescent as GD
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from utility import *


def main():
    
    lastNorm1 = 0
    lastIterations = 0
    lastTime = 0
    plt.figure(figsize=(17, 13))
    ax1 = plt.subplot(4, 1, 1)

    for i in range(10):

        A = readMatrix('F',i+1)
        f = nf.normFunction(A)
        optimizer = GD.steepestGradientDescent(f)

        time_start = time.perf_counter()
        norm1, iterations = optimizer.steepestGradientDescent()
        time_elapsed = (time.perf_counter() - time_start)

        if(i!=0):
            ax1.plot([lastIterations,iterations],[lastTime,time_elapsed], 'o-')
            ax1.set_ylabel('Times')
            ax1.set_xlabel('Iterations')
            # ax1.plot([lastIterations,iterations],[lastTime,time_elapsed], 'o-')

        lastNorm1 = norm1
        lastIterations = iterations
        lastTime = time_elapsed

    # Per Serie e Singola
    plt.show()


if __name__ == "__main__":
    main()
