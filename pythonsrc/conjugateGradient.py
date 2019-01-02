from numpy import linalg as LA
import numpy as np
import conf

class conjugateGradient():
    def __init__(self, function, x = None):
        self.function = function
        self.feval = 1
        self.x = x if x != None else self.function.init_x()
        self.v, self.g = function.calculate(self.x)
        self.g = -self.g
        self.ng = LA.norm(self.g)
        self.pOld = -1
        self.p = -self.g
        self.B = 0
        self.gTg = np.matmul(self.g.T, self.g)
        # Absolute error or relative error?
        if conf.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1

    def ConjugateGradient(self):
        while True:
            self.print()
            # Norm of the gradient lower or equal of the epsilon
            if self.ng <= conf.eps * self.ng0:
                self.status = 'optimal'
                break

            # Man number of iteration?
            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                break

            # calculate step along direction
            alpha = self.function.conjugateGradientStepsize(self.p)
            # step too short
            if alpha <= conf.mina:
                self.status = 'error'
                break

            self.oldgTg  = self.gTg
            ###########
            self.x = self.x + alpha * self.g
            ###########
            self.v, self.g = self.function.calculate(self.x)
            self.g = -self.g
            self.feval = self.feval + 1
            self.gTg = np.matmul(self.g.T, self.g)
            self.B = self.gTg/self.oldgTg
            self.pOld = self.p
            self.p = -self.g + ((self.pOld)*self.B)
            

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                break
            self.ng = LA.norm(self.g)

    def print(self):
        print("Iterations number %d, -f(x) = %0.4f, gradientNorm = %f"%( self.feval, self.v, self.ng))