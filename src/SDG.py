from numpy import *
from myFunction import *

#Leggere changelog
class GradDescent():

    def __init__(self, function, x = None):

        self.mina = 1e-16
        self.sfgrd = 0.01
        self.eps = 1e-6
        self.MaxFeval = 1000
        self.MInf = - float("inf")
        self.function = function
        '''
        self.g = nabla
        self.x = x
        self.A = matrix

        print(self.x)
        print(self.v)
        print(self.fStar)
        print(self.g)
        '''

        # Variabili globali

        self.lastx = zeros((n, 1))  # last point visited in the line search
        self.lastg = zeros((n, 1))  # gradient of lastx
        self.feval = 1  # f() evaluations count ("common" with LSs)

        # Inizializzazione
        
        print('Gradient Method \n')
        '''
        if self.fStar > - float("inf"):
            print('feval\trel gap\t\t|| g(x) ||\trate\t\tls feval\ta*')
            self.prevv = - float("inf")
        else:
            print('feval\tf(x)\t\t\t|| g(x) ||')
        print('\n\n')
        '''
        self.x = x if x !=None else self.init_x()
        self.v, self.g = function.calculate(x)
        self.ng = linalg.norm(self.g)
        
        if self.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1
        

    def SDGLoop(self):

        while True:
            self.print()
            if self.ng <= self.eps * self.ng0:
                self.status = 'optimal'
                break

            if self.feval > self.MaxFeval:
                self.status = 'stopped'
                break
            # calculate step along direction
            alpha = self.function.stepsize()
            if alpha <= self.mina:
                self.status = 'error'
                break
            self.x = self.x - alpha * self.g
            self.v, self.g = self.function(self.x)
            self.feval = self.feval + 1
            if self.v <= self.MInf:
                self.status = 'unbounded'
                break
            self.ng = linalg.norm(self.g)

            '''
            if self.fStar > - float("inf"):

                print(str(self.feval) + '\t' + str((self.v - self.fStar) / np.maximum(abs(self.fStar), 1)) + '\t' + str(self.ng),)

                if self.prevv < float("inf"):
                    print('\t' + str((self.v - self.fStar) / (self.prevv - self.fStar)),)
                else:
                    print('\t\t',)

                self.prevv = self.v
            else:
                print('ci entro')
                print(str(self.feval) + '\t' + str(self.v) + '\t\t' + str(self.ng),)

            if self.ng <= self.eps * self.ng0:
                self.status = 'optimal'
                break

            

            MaxFeval = 1000
            m1 = 0.01
            m2 = 0.9
            astart = 1
            tau = 0.9
            MInf = - float("inf")

            if self.feval > MaxFeval:
                self.status = 'stopped'
                break
            


            # compute step size

            phip0 = - self.ng * self.ng
            # a , self.v = self.ArmijoWolfesLS(self.v, phip0, astart, m1, m2, tau)

            # output statistics

            print('\t' + str(a),)
            print('\n')

            if a <= self.mina:
                self.status = 'error'
                break

            if self.v <= MInf:
                self.status = 'unbounded'
                break

            self.x = self.lastx

            self.g = self.lastg
            self.ng = linalg.norm(self.g)
            '''

        print('\nRisultato = ' + str(self.x))

    def print(self):
        print("Iterations number %d, f(x) = %0.4f, gradientNorm = %0.4f"%(self.feval, self.v, self.ng))

    # cool but we should not use it I guess
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

        print(lister)
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
        print('lister', lister)
        return a, phia

    def f2phi(self, alpha):

        self.lastx = self.x - alpha * self.g
        phi, self.lastg, tmp3 = function.calculate(self.lastx)
        phip = - self.g.conj().transpose() * self.lastg
        self.feval = self.feval + 1

        return phi, phip
