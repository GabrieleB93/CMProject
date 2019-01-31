import numpy as np

#
# This file implements the norm function
# which is the objective function that we will opimize
# with the conjugate gradient and the steepestDescentDirection
# It has an API as the two algorithm wants
#
class normFunction():
    def __init__(self, A):
        self.A = A
        # Q = A'A 
        self.Q = np.matmul(np.transpose(A), A)
        self.dim = self.Q.shape[0]

    # fucntion that calculate f(x) and grad(x)
    def calculate(self, x):
        # f(x) = x'Qx / x'x
        self.x = x
        self.xT = x.T
        self.xTx = np.matmul(self.xT, x)
        self.Qx = np.matmul(self.Q, x)
        self.xQx = np.matmul(self.xT, self.Qx)
        f_x = self.xQx / self.xTx
        nabla_f = (2 * x * f_x) / self.xTx - (2 * self.Qx) / self.xTx
        self.f_x = f_x  # it's -f(x)
        self.d = nabla_f
        self.dT = self.d.T
        return f_x, nabla_f

    # funciton that return the step size of the algorithm using exact line search along the 
    # graient direction. It uses the closed formula of the derivative along the direction
    def stepsizeAlongGradientDirection(self):
        # expressed the poly as aalpha^2+b*alpha+c 
        # we will have 
        # a = (d.T*d)(x*Q*d) - (d.T*Q*d)*(x.T*d)  
        dTd = np.matmul(self.dT, self.d)
        xTd = np.matmul(self.xT, self.d)
        self.xTx = np.matmul(self.xT, self.x)
        Qd = np.matmul(self.Q, self.d)
        xQd = np.matmul(self.xT, Qd)
        dQd = np.matmul(self.dT, Qd)
        a = float(dTd * xQd - dQd * xTd)
        # b = (xTx)(dQd) - (dTd)(xQx)
        b = float(self.xTx * dQd - dTd * self.xQx)
        # c = (xTd)(xQx) - (xTx)(xQd)
        c = float(xTd * self.xQx - self.xTx * xQd)
        # now alpha is the solution of ax^2+bx+c : x > 0 
        coef = np.array([a, b, c])
        roots = np.roots(coef)
        if roots[0] < 0 and roots[1] < 0:
            return 0
        elif roots[0] < 0:
            return roots[1]
        elif roots[1] < 0:
            return roots[0]
        return np.min([roots])

    def exactSearchGeneralDirection(self, d):
        # expressed the poly as aalpha^2+b*alpha+c 
        # we will have 
        # a = (d.T*d)(x*Q*d) - (d.T*Q*d)*(x.T*d)  
        dTd = np.matmul(d.T, d)
        xTd = np.matmul(self.xT, d)
        self.xTx = np.matmul(self.xT, self.x)
        Qd = np.matmul(self.Q, d)
        xQd = np.matmul(self.xT, Qd)
        dQd = np.matmul(d.T, Qd)
        a = float(dTd * xQd - dQd * xTd)
        # b = (xTx)(dQd) - (dTd)(xQx)
        b = float(self.xTx * dQd - dTd * self.xQx)
        # c = (xTd)(xQx) - (xTx)(xQd)
        c = float(xTd * self.xQx - self.xTx * xQd)
        # now alpha is the solution of ax^2+bx+c : x > 0 
        coef = np.array([a, b, c])
        roots = np.roots(coef)
        if roots[0] < 0 and roots[1] < 0:
            return 0
        elif roots[0] < 0:
            return roots[1]
        elif roots[1] < 0:
            return roots[0]
        return np.min([roots])

    def init_x(self):
        return np.random.rand(self.dim, 1)
