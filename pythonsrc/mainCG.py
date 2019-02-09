import normFunction as nf
import conjugateGradient as CG
import numpy as np
from numpy import linalg as LA
from utility import *

def main(A = None):

    type = "C"
    relerrorsCG = []
    gradientsCG = []

    A = readMatrix(type, 1)
    f = nf.normFunction(A)
    initial_vector = f.init_x()

    # Optimizer CG
    optimizerCG = CG.conjugateGradient(f, initial_vector, True)
    gradientCG, normsCG = optimizerCG.ConjugateGradient()

    # Norm numpy
    norm = LA.norm(A, ord=2) ** 2

    # Norm and errors CG
    normsCG = np.array(normsCG)
    gradientsCG.insert(0, np.array(gradientCG))
    size1 = normsCG.size

    normvec = np.ones(size1) * norm
    relerrorsCG.insert(0, (abs(normsCG - normvec) / abs(normvec)))

    printPlot2(None, None, relerrorsCG, gradientsCG, A, type, "1")


if __name__ == "__main__":
    main()
