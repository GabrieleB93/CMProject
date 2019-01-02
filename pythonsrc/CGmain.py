import normFunction as nf
import conjugateGradient as CG
import numpy as np
from numpy import linalg as LA


def main():
    A = np.matrix('1 4 5 5; 1 4 5 122; 12 6 4 1')
    f = nf.normFunction(A)
    optimizer = CG.conjugateGradient(f)
    optimizer.ConjugateGradient()
    norm = LA.norm(A, ord = 2)
    print(norm**2)

if __name__ == "__main__":
    main()
