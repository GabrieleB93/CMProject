import normFunction as nf
import steepestGradientDescent as SGD
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from utility import *

def main():
    A = readMatrix('A',1)
    f = nf.normFunction(A)
    optimizer = SGD.steepestGradientDescent(f)
    gradients, norms = optimizer.steepestGradientDescent()
    norms = np.array(norms)
    gradients = np.array(gradients)
    norm = LA.norm(A, ord = 2)**2
    size = norms.size
    normvec = np.ones(size)*norm
    errors = np.log10(abs(norms - normvec))
    relerrors = np.log10(abs(norms - normvec)/norm)
    for i in range(size):
        if errors[i] == float("-inf"):
            errors[i] = -16
    fig = plt.subplot(2,1,1)
    fig.plot(errors)
    fig = plt.subplot(2,1,2)
    fig.plot(np.log10(gradients))
    plt.show()
if __name__ == "__main__":
    main()
