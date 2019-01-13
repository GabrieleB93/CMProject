import numpy as np

def readMatrix(type,number):
    raw = []
    with open('Matrices/Matrix'+type+'/matrix'+type+str(number)+'.txt', 'r') as f:
        for line in f:
            raw.append(line.split())
    A = np.array(raw, dtype=float)
    return A
