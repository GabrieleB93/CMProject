from numpy import *
from myFunction import *


class GradDescent:

    def __init__(self, function, nabla, x, matrix, q):

        self.mina = 1e-16
        self.sfgrd = 0.01
        self.eps = 1e-6
        n = nabla.shape[1]

        #Temporane0
        self.q = q

        tmp = MyFunction(matrix)
        self.fStar, tmp1, tmp2 = tmp.calculate(np.matrix(''), self.q)
        self.v = function
        self.g = nabla
        self.x = x
        self.A = matrix

        print self.x
        print self.v
        print self.fStar
        print self.g


        # Variabili globali

        self.lastx = zeros((n, 1))  # last point visited in the line search
        self.lastg = zeros((n, 1))  # gradient of lastx
        self.feval = 1  # f() evaluations count ("common" with LSs)

        print 'Gradient Method \n'
        if self.fStar > - float("inf"):
            print 'feval\trel gap\t\t|| g(x) ||\trate\t\tls feval\ta*'
            self.prevv = - float("inf")
        else:
            print 'feval\tf(x)\t\t\t|| g(x) ||'
        # print '\tls feval\ta*'
        print '\n\n'

        # f = MyFunction(self.A)
        # self.v, self.g = f.calculate(self.x, np.matrix('10;5'))

        self.ng = linalg.norm(self.x)

        if self.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1

    def SDGLoop(self):

        while True:
            if self.fStar > - float("inf"):

                print str(self.feval) + '\t' + str((self.v - self.fStar) / np.maximum(abs(self.fStar), 1)) + '\t' + str(
                    self.ng),

                if self.prevv < float("inf"):
                    print '\t' + str((self.v - self.fStar) / (self.prevv - self.fStar)),
                else:
                    print '\t\t',

                self.prevv = self.v
            else:
                print 'ci entro'
                print  str(self.feval) + '\t' + str(self.v) + '\t\t' + str(self.ng),

            if self.ng <= self.eps * self.ng0:
                status = 'optimal'
                break

            MaxFeval = 1000
            m1 = 0.01
            m2 = 0.9
            astart = 1
            tau = 0.9
            MInf = - float("inf")

            if self.feval > MaxFeval:
                status = 'stopped'
                break

            # compute step size

            phip0 = - self.ng * self.ng
            a , self.v = self.ArmijoWolfesLS(self.v, phip0, astart, m1, m2, tau)

            # output statistics

            print '\t' + str(a),
            print '\n'

            if a <= self.mina:
                status = 'error'
                break

            if self.v <= MInf:
                status = 'unbounded'
                break

            self.x = self.lastx

            self.g = self.lastg
            self.ng = linalg.norm(self.g)

        print '\nRisultato = ' + str(self.x)

    def ArmijoWolfesLS(self, phi0, phip0, as_, m1, m2, tau):

        global phia
        lister = 1  # count iterations of first phase

        while self.feval <= 1000:

            phia, self.phips = self.f2phi(as_)

            if (phia <= (phi0 + m1 * as_ * phip0)) and abs(self.phips) <= (-m2 * phip0):
                print lister
                a = as_
                return a, phia # Armijo  +strong Wolfe satisfied, we are done

            if self.phips >= 0:
                break

            as_ = as_ / tau
            lister = lister + 1

        print lister
        lister = 1  # count iteration of second phase

        am = 0
        a = as_
        phipm = phip0

        while (self.feval <= 1000) and ((as_ - am) > self.mina) and (self.phips > 1e-12):

            a = (am * self.phips - as_ * phipm) / (self.phips - phipm)
            a = np.maximum((am * (1 + self.sfgrd)), np.minimum((as_ * (1 - self.sfgrd)), a))

            phia, phip = self.f2phi(a.item())

            if phia <= phi0 + m1 * a * phi0 and abs(phip) <= -m2 * phip0:
                break  # Armijo + Strong Wolfe satisfied

            if phip < 0:
                am = a
                phipm = phip
            else:
                as_ = a

                if as_ <= self.mina:
                    break

            self.phips = phip

            lister = lister + 1
        print 'lister', lister
        return a, phia

    def f2phi(self, alpha):

        self.lastx = self.x - alpha * self.g
        f = MyFunction(self.A)
        phi, self.lastg, tmp3 = f.calculate(self.lastx,self.q)
        phip = - self.g.conj().transpose() * self.lastg
        self.feval = self.feval + 1

        return phi, phip
