import numpy as np
import conjugateGradient as CG
import steepestGradientDescent as SGD
from numpy import linalg as LA
import normFunction as nf
from utility import *

espereiments = 100
numberOfMatrix = 10
typeMatrix = "A"
typeMatrixNum = 64
maxType = 8


def main():
    global A

    for i in range(1, maxType + 1):

        relerrorsSGD = []
        gradientsSGD = []
        relerrorsCG = []
        gradientsCG = []

        type = (chr(typeMatrixNum + i))

        for j in range(1, numberOfMatrix + 1):
            A = readMatrix(type, j)
            f = nf.normFunction(A)
            initial_vector = f.init_x()

            # Optimizer SGD and CG
            optimizerSGD = SGD.steepestGradientDescent(f, initial_vector, False)
            optimizerCG = CG.conjugateGradient(f, initial_vector, False)
            gradientCG, normsCG = optimizerCG.ConjugateGradient()
            gradientSGD, normsSGD = optimizerSGD.steepestGradientDescent()

            # Norm numpy
            norm = LA.norm(A, ord=2) ** 2

            # Norm and errors SGD
            normsSGD = np.array(normsSGD)
            gradientsSGD.insert(j - 1, np.array(gradientSGD))
            size1 = normsSGD.size

            normvec = np.ones(size1) * norm
            # errorsSGD.insert(j - 1, abs(normsSGD - normvec))
            relerrorsSGD.insert(j - 1, abs(normsSGD - normvec) / abs(normvec))

            # Norm and errors CG
            normsCG = np.array(normsCG)
            gradientsCG.insert(j - 1, np.array(gradientCG))

            size2 = normsCG.size
            normvec = np.ones(size2) * norm
            relerrorsCG.insert(j - 1, abs(normsCG - normvec) / abs(normvec))

        # printPlot(None, relerrorsSGD, gradientsSGD, None, relerrorsCG, gradientsCG, A, type, str(i))
        printPlot2(relerrorsSGD, gradientsSGD, relerrorsCG, gradientsCG, A, type, str(i))

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))

if __name__ == "__main__":
    main()
