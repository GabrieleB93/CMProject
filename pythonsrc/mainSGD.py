import normFunction as nf
import steepestGradientDescent as SGD
import numpy as np
from numpy import linalg as LA
from utility import *

def main():

    type = "A"
    relerrorsSGD = []
    gradientsSGD = []

    A = readMatrix(type, 1)
    f = nf.normFunction(A)
    initial_vector = f.init_x()

    # Optimizer SGD
    optimizerSGD = SGD.steepestGradientDescent(f, initial_vector, True)
    gradientSGD, normsSGD = optimizerSGD.steepestGradientDescent()

    # Norm numpy
    norm = LA.norm(A, ord=2) ** 2

    # Norm and errors SGD
    normsSGD = np.array(normsSGD)
    gradientsSGD.insert(0,np.array(gradientSGD))
    size1 = normsSGD.size

    normvec = np.ones(size1) * norm
    relerrorsSGD.insert(0, (abs(normsSGD - normvec) / abs(normvec)))

    printPlot2(relerrorsSGD, gradientsSGD, None, None, A, type, "1")


if __name__ == "__main__":
    main()
