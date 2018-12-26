import numpy as np

class myFunction():
    def __init__(self, A):
        self.A = A
        # Q = A'A
        self.Q = np.matmul(np.transpose(A), A)

    # fucntion that calculate f(x) and grad(x)
    def calculate(self, x):
        # f(x) = x'Qx / x'x
        f_x = np.matmul(np.matmul(np.transpose(x), self.Q), x)/np.matmul(np.transpose(x), x)
        # grad(x) = 
        gradx = np.matmul()# da scrivere
        return f_x, gradx
