import conf
import numpy as np
from numpy import linalg as LA

#
# This file implements the optimization of a function with the gradient method. 
# The class steepestGradientDescent needs a function object whith this three methods: 
# 1) init_x() -> return the starting point (could be usefull for some fucntions)
# 2) calculate(ponitX) -> returns f(X) and gradient in X 
# 3) stepsizeAlongGradientDirection(direction) -> return a point that satisfy at least the Armijo-Wolf condition along the gradient direction
#  
class steepestGradientDescent():
    def __init__(self, function, x=None, verbose = True):
        self.verbose = verbose
        self.function = function
        self.status = ''
        self.feval = 1
        self.x = x if x is not None else self.function.init_x()

        self.v, self.g = function.calculate(self.x)
        self.ng = LA.norm(self.g)
        # Absolute error or relative error?
        if conf.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1

    def steepestGradientDescent(self):
        self.historyNorm = []
        self.historyValue = []
        while True:
            self.historyNorm.append(np.asscalar(self.ng))
            self.historyValue.append(np.asscalar(self.v))
            if self.verbose: 
                self.print()

            # Norm of the gradient lower or equal of the epsilon
            if self.ng <= conf.eps * self.ng0:
                self.status = 'optimal'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue


            # Man number of iteration?
            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            # calculate step along direction
            alpha = self.function.stepsizeAlongGradientDirection()

            # step too short
            if alpha <= conf.mina:
                self.status = 'error'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            lastx = self.x
            self.x = self.x - alpha * self.g
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            self.ng = LA.norm(self.g)

        print('\n x = ' + str(self.x) + '\nvalue = %0.4f' % self.v)

    def print(self):
        print("Iterations number %d, -f(x) = %0.4f, gradientNorm = %f - " % (self.feval, self.v, self.ng) + self.status)

    # same function as the previus one but it returns also the time and
    # we avoid print and other operation which slow down the algorithm
    def steepestGradientDescentTIME(self):
        while True:
            if self.ng <= conf.eps * self.ng0:
                self.status = 'optimal'
                return np.asscalar(self.ng), np.asscalar(self.v)

            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                #print(self.status)
                break

            alpha = self.function.stepsizeAlongGradientDirection()

            # step too short
            if alpha <= conf.mina:
                self.status = 'error'
                #print(self.status)
                break

            self.x = self.x - alpha * self.g
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                #print(self.status)
                break

            self.ng = LA.norm(self.g)
        return np.asscalar(self.ng), np.asscalar(self.v)
