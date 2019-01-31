from numpy import linalg as LA
import numpy as np
import conf
import matplotlib.pyplot as plt
import time

#
# This file implements the optimization of a function with the conjugte gradient method. 
# The class conjugateGradient needs a function object whith this three methods: 
# 1) init_x() -> return the starting point (could be usefull for some fucntions)
# 2) calculate(ponitX) -> returns f(X) and gradient in X 
# 3) exactSearchDirection(direction) -> return a point that satisfy at least the Armijo-Wolf condition
#  
class conjugateGradient():
    # init function
    def __init__(self, function, x = None, verbose = True):
        self.verbose = verbose
        self.function = function
        # it stores the number of iterations
        self.feval = 1
        # x_0
        self.x = x if x is not None else self.function.init_x()
        # These objects are used by the CG algorithm
        # self.v = f(X)
        # self.g = \nabla f(X)
        
        self.v, self.g = function.calculate(self.x)
        # self.pOld = old value of p (see CG if you want to know what is p)
        # self.p = new value of p
        self.pOld = -1
        self.p = -self.g
        # self.B = beta -> we use the Fletcher–Reeves beta, but we can easly implements other beta
        self.B = 0
        # self.gTg = norm2(g)**2
        self.gTg = np.matmul(self.g.T, self.g)
        # if conf.eps < 0 we will use the relative error as a stopping criteria
        if conf.eps < 0:
            self.ng0 = - np.sqrt(self.gTg)
        else:
            self.ng0 = 1

    def ConjugateGradient(self):
        # object for storing the steps
        self.historyNorm = []
        self.historyValue = []
        while True:
            self.historyNorm.append(np.asscalar(np.sqrt(self.gTg)))
            self.historyValue.append(np.asscalar(self.v))
            if self.verbose:
                self.print()
            # Norm of the gradient lower or equal of the epsilon-> first stopping criteria
            if np.sqrt(self.gTg) <= conf.eps * self.ng0:
                self.status = 'optimal'
                return self.historyNorm, self.historyValue

            # If we reach the maximum number of iteration we have to stop -> second stopping criteria
            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                return self.historyNorm, self.historyValue

            # calculate step along direction
            # -direction because model calculate the derivative of 
            # phi' = f'(x-aplha*d)
            alpha = self.function.exactSearchDirection(-self.p)

            # if the stop is too short we stop
            if alpha <= conf.mina:
                self.status = 'error'
                return self.historyNorm, self.historyValue
            
            # now we will update all the CG variables
            self.oldgTg  = self.gTg
            lastx = self.x
            # update x 
            self.x = self.x + alpha * self.p
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1
            self.gTg = np.matmul(self.g.T, self.g)
            self.B = self.gTg/self.oldgTg
            self.pOld = self.p
            self.p = -self.g + ((self.pOld)*self.B)

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                return self.historyNorm, self.historyValue
                # break

    def print(self):
        print("Iterations number %d, -f(x) = %0.4f, gradientNorm = %f"%( self.feval, self.v, np.sqrt(self.gTg)))

    def ConjugateGradientTIME(self):
        while True:
            # Norm of the gradient lower or equal of the epsilon
            if np.sqrt(self.gTg) <= conf.eps * self.ng0:
                self.status = 'optimal'
                return np.asscalar(self.gTg), np.asscalar(self.v)

            # Man number of iteration?
            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                break

            # calculate step along direction
            # -direction because model calculate the derivative of 
            # phi' = f'(x-aplha*d)
            alpha = self.function.exactSearchDirection(-self.p)
            # step too short
            if alpha <= conf.mina:
                self.status = 'error'
                break
            

            self.oldgTg  = self.gTg
            lastx = self.x
            self.x = self.x + alpha * self.p
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1
            self.gTg = np.matmul(self.g.T, self.g)
            self.B = self.gTg /self.oldgTg
            self.pOld = self.p
            self.p = -self.g + ((self.pOld)*self.B)

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                break

        return np.asscalar(self.gTg), np.asscalar(self.v)