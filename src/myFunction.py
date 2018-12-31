import numpy as np
from numpy import linalg


class MyFunction():
    def __init__(self, A):

        self.A = A
        # Q = A'A
        # self.Q = np.matmul(np.transpose(A), A)

        #Temporaneo, per esempio del prof
        self.Q = self.A

    # fucntion that calculate f(x) and grad(x)
    def calculate(self, x, q):

        #Ricordarsi di togliere q, per il nostro caso

        # f(x) = x'Qx / x'x

        if not x.size:

            if np.min(linalg.eig(self.Q)[0]) > 1e-14:

                #Per la nostra funzione
                # print np.ones((self.Q.shape[1],1))
                # xStar = linalg.solve(self.Q, - np.ones((self.Q.shape[1],1)))

                # xTx = np.matmul(np.transpose(xStar), xStar)
                # f_x = np.matmul(np.matmul(np.transpose(xStar), self.Q), xStar) / xTx

                #Esempio Prof
                xStar = linalg.solve(self.Q, - q)
                f_x = 0.5 * xStar.transpose() * self.Q * xStar + q.transpose() * xStar


            else:
                f_x = - float("inf")

            x = np.matrix(' 0 ; 0')
            nabla_f = np.matrix(' 0 ; 0') #Rivedere

        else:

            #
            # xTx = np.matmul(np.transpose(x), x)
            # f_x = np.matmul(np.matmul(np.transpose(x), self.Q), x) / xTx
            # nabla_f = (2 * x * f_x) / xTx - (np.matmul(2 * self.Q, x)) / xTx

            f_x = 0.5 * x.transpose() * self.Q * x + q.transpose() * x
            nabla_f = self.Q * x + q

        return f_x, nabla_f , x
