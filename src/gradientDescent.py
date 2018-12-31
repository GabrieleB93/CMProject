
from numpy import linalg as LA
import numpy as np

class gradDescent():
    def __init__(self, function, x = None):
        self.mina = 1e-16
        self.sfgrd = 0.01
        self.eps = 1e-6
        self.MaxFeval = 1000
        self.MInf = - float("inf")
        self.function = function
        self.feval = 1
        self.x = x if x !=None else self.function.init_x()
        self.v, self.g = function.calculate(self.x)
        self.ng = LA.norm(self.g)
        # Absolute error or relative error?
        if self.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1
    
    def SDG(self):
        while True:
            self.print()
            # Norm of the gradient lower or equal of the epsilon
            if self.ng <= self.eps * self.ng0:
                self.status = 'optimal'
                break

            # Man number of iteration?
            if self.feval > self.MaxFeval:
                self.status = 'stopped'
                break

            # calculate step along direction
            alpha = self.function.stepsize()

            # step too short
            if alpha <= self.mina:
                self.status = 'error'
                break

            self.x = self.x - alpha * self.g
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1

            if self.v <= self.MInf:
                self.status = 'unbounded'
                break
            self.ng = LA.norm(self.g)

        print('\n x = ' + str(self.x)+'\nvalue = %0.4f' %self.v)


    def print(self):
        print("Iterations number %d, f(x) = %0.4f, gradientNorm = %f"%(self.feval, self.v, self.ng))