from numpy import linalg as LA
import numpy as np
import conf
import matplotlib.pyplot as plt


class gradDescent():
    def __init__(self, function, x=None):
        self.function = function
        self.feval = 1
        self.x = x if x != None else self.function.init_x()
        self.v, self.g = function.calculate(self.x)
        self.ng = LA.norm(self.g)
        # Absolute error or relative error?
        if conf.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1

    def SDG(self, indx=None):

        if(indx!=None):
            if (indx > 3):
                plot = plt.subplot(4, 3, indx)

        while True:
            self.print() #Non sarebbe più corretto spostarlo dopo l' IF? Perchè attualmente
            # viene segnata un'iterazione in più, o no?

            # Norm of the gradient lower or equal of the epsilon
            if self.ng <= conf.eps * self.ng0:
                self.status = 'optimal'
                return self.v, self.feval
                # break

            # Man number of iteration?
            if self.feval > conf.MaxFeval:
                self.status = 'stopped'
                break

            # calculate step along direction
            alpha = self.function.stepsize()

            # step too short
            if alpha <= conf.mina:
                self.status = 'error'
                break

            lastx = self.x
            self.x = self.x - alpha * self.g
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1

            # Per il plot da P1(x,y) a P2(x,y)
            if(indx!=None):
                if (indx > 3):
                    plot.plot([self.x.item(0), lastx.item(0)], [self.x.item(1), lastx.item(1)], 'o-')
            else:
                plt.plot([self.x.item(0), lastx.item(0)], [self.x.item(1), lastx.item(1)], 'o-')

            if self.v <= conf.MInf:
                self.status = 'unbounded'
                break
            self.ng = LA.norm(self.g)

        print('\n x = ' + str(self.x)+'\nvalue = %0.4f' %self.v)

    def print(self):
        print("Iterations number %d, -f(x) = %0.4f, gradientNorm = %f" % (self.feval, self.v, self.ng))
