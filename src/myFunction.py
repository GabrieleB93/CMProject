import numpy as np

class myFunction():
    def __init__(self, A):
        self.A = A
        # Q = A'A
        self.Q = np.matmul(np.transpose(A), A)

    # fucntion that calculate f(x) and grad(x)
    def calculate(self, x):
        # f(x) = x'Qx / x'x
        xTx = np.matmul(np.transpose(x), x)
        f_x = np.matmul(np.matmul(np.transpose(x), self.Q), x)/xTx
        nabla_f = (2*x*f_x)/xTx - (np.matmul(2*self.Q, x))/xTx 
        return f_x, nabla_f
