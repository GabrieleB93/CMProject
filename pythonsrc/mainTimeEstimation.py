import numpy as np
import conjugateGradient as CG
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from utility import *

espereiments = 100
numberOfMatrix = 10
typeMatrix = "G"

def main():
    print("matrix type "+typeMatrix)
    for i in range(1,numberOfMatrix+1):
        A = readMatrix(typeMatrix,i)
        CGtimes = []
        SGDtimes = []
        numpytimes = []
        for i in range(espereiments):
            # conjugate gradient
            startime = time.time()
            _, norm = normCG(A)
            CGtimes.append(time.time()- startime)
            # steepest gradient descent
            startime = time.time()
            _, norm = normSDG(A)
            SGDtimes.append(time.time()- startime)
            # numpy norm
            startime = time.time()
            norm = LA.norm(A, ord = 2)**2
            numpytimes.append(time.time()- startime)
        CGtimes = np.array(CGtimes)
        SGDtimes = np.array(SGDtimes)
        numpytimes = np.array(numpytimes)
    print("Conjugate gradient time %.6f \nSteepest gradient descent average time %.6f \nNumpy ninealog average time %.6f" %(np.mean(CGtimes),np.mean(SGDtimes),np.mean(numpytimes)))



if __name__ == "__main__":
    main()