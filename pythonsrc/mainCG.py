import normFunction as nf
import conjugateGradient as CG
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from utility import *

def main():
    A = readMatrix('A',1)
    f = nf.normFunction(A)

    optimizer = CG.conjugateGradient(f)
    gradients, norms = optimizer.ConjugateGradient()
    norms = np.array(norms)
    gradients = np.array(gradients)
    norm = LA.norm(A, ord = 2)**2
    size = norms.size
    normvec = np.ones(size)*norm
    errors = np.log10(abs(norms - normvec))
    for i in range(size):
        if errors[i] == float("-inf"):
            errors[i] = -16
    fig = plt.subplot(2,1,1)
    fig.plot(errors)
    fig = plt.subplot(2,1,2)
    fig.plot(np.log10(gradients))
    plt.show()
    #print("Norm = %.16f \nnorm2 = %.16f\ndiff = %.30f" % (norm[-1], norms[-1], errors[-1]))

if __name__ == "__main__":
    main()
