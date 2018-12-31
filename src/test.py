from SDG import *
import numpy as np

#Matrici e vettori uguali a genericquad, dall'esempio del prof
#Decommentato = la funzione quadratica del prof
#Commentata = la nostra funzione

m = np.matrix('6 -2; -2 6')
x = np.matrix('1;1')

#Superfluo nel nostro caso
q = np.matrix('10;5')

test = MyFunction(m)
function, nabla , x_0 = test.calculate(x,q)

sdg = GradDescent(function, nabla, x_0, m, q)
sdg.SDGLoop()

#Per il secondo caso, vettore x vuoto, da controllare
# m1 = np.matrix('1 2 3; 1 2 3')
# x1 = np.matrix('')
#
# test1 = MyFunction(m1)
# test1.calculate(x1)
