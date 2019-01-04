import normFunction as nf
import gradientDescent as GD
import numpy as np
from numpy import linalg as LA


def main():
    A = np.matrix('1 4 5 5; 1 4 5 122; 12 6 4 1')
    f = nf.normFunction(A)
    optimizer = GD.gradDescent(f)
    norm1 = optimizer.SDG()
    norm = LA.norm(A, ord = 2)
    print("Norm = %.16f \n norm2 = %.16f\n diff = %.30f" %(norm**2, norm1,  np.log10(abs(norm1-norm**2))))

if __name__ == "__main__":
    main()

