import steepestGradientDescent as SGD
import conjugateGradient as CG
import numpy as np
import normFunction as nf

def readMatrix(type,number):
    raw = []
    with open('Matrices/Matrix'+type+'/matrix'+type+str(number)+'.txt', 'r') as f:
        for line in f:
            raw.append(line.split())
    A = np.array(raw, dtype=float)
    return A

# we created this two functions so we can calculate 
# the norm as a call froma an API
def normCG(A):
    f = nf.normFunction(A)
    CGoptimizer = CG.conjugateGradient(f)
    return CGoptimizer.ConjugateGradientTIME()

def normSDG(A):
    f = nf.normFunction(A)
    SGDoptimizer = SGD.steepestGradientDescent(f)
    return SGDoptimizer.steepestGradientDescentTIME()