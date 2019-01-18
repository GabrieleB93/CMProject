from numpy import linalg as LA
import numpy as np
import conf
import matplotlib.pyplot as plt
import time

class conjugateGradient():
    def __init__(self, function, x = None, verbose = True):
        self.verbose = verbose
        self.function = function
        self.feval = 1
        self.x = x if x is not None else self.function.init_x()
        self.v, self.g = function.calculate(self.x)
        self.pOld = -1
        self.p = -self.g
        self.B = 0
        self.ng = np.matmul(self.g.T, self.g)
        # Absolute error or relative error?
        if conf.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1

    def ConjugateGradient(self):
        self.historyNorm = []
        self.historyValue = []
        while True:
            self.historyNorm.append(np.asscalar(self.ng))
            self.historyValue.append(np.asscalar(self.v))
            if self.verbose:
                self.print()
            # Norm of the gradient lower or equal of the epsilon
            if np.sqrt(self.ng) <= conf.eps * self.ng0:
                self.status = 'optimal'
                return self.historyNorm, self.historyValue

            # Man number of iteration?
            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                return self.historyNorm, self.historyValue
                # break

            # calculate step along direction
            # -direction because model calculate the derivative of 
            # phi' = f'(x-aplha*d)
            alpha = self.function.conjugateGradientStepsize(-self.p)
            # step too short
            if alpha <= conf.mina:
                self.status = 'error'
                return self.historyNorm, self.historyValue
                # break
            

            self.oldgTg  = self.ng
            ###########
            lastx = self.x
            self.x = self.x + alpha * self.p
            ###########
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1
            self.ng = np.matmul(self.g.T, self.g)
            self.B = self.ng/self.oldgTg
            self.pOld = self.p
            self.p = -self.g + ((self.pOld)*self.B)

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                return self.historyNorm, self.historyValue
                # break

    def print(self):
        print("Iterations number %d, -f(x) = %0.4f, gradientNorm = %f"%( self.feval, self.v, self.ng))

    def ConjugateGradientTIME(self):
        while True:
            # Norm of the gradient lower or equal of the epsilon
            if np.sqrt(self.ng) <= conf.eps * self.ng0:
                self.status = 'optimal'
                return np.asscalar(self.ng), np.asscalar(self.v)

            # Man number of iteration?
            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                break

            # calculate step along direction
            # -direction because model calculate the derivative of 
            # phi' = f'(x-aplha*d)
            alpha = self.function.conjugateGradientStepsize(-self.p)
            # step too short
            if alpha <= conf.mina:
                self.status = 'error'
                break
            

            self.oldgTg  = self.ng
            lastx = self.x
            self.x = self.x + alpha * self.p
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1
            self.ng = np.matmul(self.g.T, self.g)
            self.B = self.ng /self.oldgTg
            self.pOld = self.p
            self.p = -self.g + ((self.pOld)*self.B)

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                break

        return np.asscalar(self.ng), np.asscalar(self.v)