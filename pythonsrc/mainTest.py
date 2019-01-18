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

        # errorsSGD = []
        relerrorsSGD = []
        gradientsSGD = []
        # errorsCG = []
        relerrorsCG = []
        gradientsCG = []

        # AVGerrorsSGD = []
        AVGrelerrorsSGD = []
        AVGgradientsSGD = []
        # AVGerrorsCG = []
        AVGrelerrorsCG = []
        AVGgradientsCG = []

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
            # errorsCG.insert(j - 1, abs(normsCG - normvec))
            relerrorsCG.insert(j - 1, abs(normsCG - normvec) / abs(normvec))

            # for z in range(size1):
            #     if errorsSGD[j - 1][z] == float("-inf"):
            #         errorsSGD[j - 1][z] = -16
            # for t in range(size2):
            #     if errorsCG[j - 1][t] == float("-inf"):
            #         errorsCG[j - 1][t] = -16

            # printPlot(None, relerrorsSGD[j-1], gradientsSGD[j-1], None, relerrorsCG[j-1], gradientsCG[j-1], A, type,str(j))

        printPlot(None, relerrorsSGD, gradientsSGD, None, relerrorsCG, gradientsCG, A, type, str(i))

        # Plot della media
        # AVGerrorsSGD.insert(0,[np.mean(arr) for arr in errorsSGD])
        # AVGrelerrorsSGD.insert(0,[np.mean(arr1) for arr1 in relerrorsSGD])
        # AVGgradientsSGD.insert(0,[np.mean(arr2) for arr2 in gradientsSGD])
        # AVGerrorsCG.insert(0,[np.mean(arr3) for arr3 in errorsCG])
        # AVGrelerrorsCG.insert(0,[np.mean(arr4) for arr4 in relerrorsCG])
        # AVGgradientsCG.insert(0,[np.mean(arr5) for arr5 in gradientsCG])
        #
        # printPlot(None, AVGrelerrorsSGD, AVGgradientsSGD, None, AVGrelerrorsCG, AVGgradientsCG, A, type,str(0))


if __name__ == "__main__":
    main()
