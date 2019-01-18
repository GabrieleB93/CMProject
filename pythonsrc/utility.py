import steepestGradientDescent as SGD
import conjugateGradient as CG
import numpy as np
import pandas as pd
import normFunction as nf
import matplotlib.pyplot as plt
import os


def readMatrix(type, number):
    raw = []
    try:
        f = open('../Matrices/Matrix' + type + '/matrix' + type + str(number) + '.txt', 'r') 
    except:
        f = open('Matrices/Matrix' + type + '/matrix' + type + str(number) + '.txt', 'r') 
    for line in f:
        raw.append(line.split())
    A = np.array(raw, dtype=float)
    return A


# we created this two functions so we can calculate
# the norm as a call froma an API
def normCG(A):
    f = nf.normFunction(A)
    CGoptimizer = CG.conjugateGradient(f)
    return CGoptimizer.ConjugateGradientTIME()


def normSDG(A):
    f = nf.normFunction(A)
    SGDoptimizer = SGD.steepestGradientDescent(f)
    return SGDoptimizer.steepestGradientDescentTIME()


def density(type):
    if type == 'A':
        return 1
    elif type == 'B':
        return 1
    elif type == 'C':
        return 1
    elif type == 'D':
        return 0.5
    elif type == 'E':
        return 0.5
    elif type == 'F':
        return 0.25
    elif type == 'G':
        return 0.25
    elif type == 'H':
        return '0.01'


def printPlot(errorsSGD=None, relerrorsSGD=None, gradientsSGD=None, errorsCG=None, relerrorsCG=None, gradientsCG=None,
              A=None, type=None, num=None, ):

    yLabel2 = 'Relative Error'
    yLabel3 = 'Norms Gradient'
    xLabel1 = 'Iterations'

    fig = plt.figure()
    relErrPlotSGD = fig.add_subplot(2,2,1 )
    relErrPlotCG = fig.add_subplot(2,2,2,sharey = relErrPlotSGD)
    gradPlotSGD= fig.add_subplot(2,2,3)
    gradPlotCG= fig.add_subplot(2,2,4,sharey = gradPlotSGD)

    fig.set_size_inches(18.5, 10.5)
    m, n = np.shape(A)

    plt.ylim(10e-16,10e0 )
    relErrPlotSGD.set_title(
        'Steepest Gradient Descent \n Type ' + type + '     Density =  ' + str(density(type)) + '    M = ' + str(
            m) + ' N = ' + str(n))

    relErrPlotSGD.set_yscale('log')
    relErrPlotSGD.set(ylabel=yLabel2)
    for relSGD in relerrorsSGD:
        relErrPlotSGD.plot(relSGD)

    gradPlotSGD.set(ylabel=yLabel3)
    gradPlotSGD.set(xlabel=xLabel1)
    gradPlotSGD.set_yscale('log')


    relErrPlotCG.set_title(
        'Conjugate Gradient \n Type ' + type + '     Density =  ' + str(density(type)) + '    M = ' + str(
            m) + ' N = ' + str(n))

    relErrPlotCG.set_yscale('log')
    for relCG in relerrorsCG:
        relErrPlotCG.plot(relCG)

    gradPlotCG.set(xlabel=xLabel1)
    gradPlotCG.set_yscale('log')

    plt.ylim(10e-10,10e5 )

    for gradCG in gradientsCG:
        gradPlotCG.plot(gradCG)
    for gradSGD in gradientsSGD:
        gradPlotSGD.plot(gradSGD)
    plt.show()
    savePlot(type, num, fig)


def savePlot(type, num, fig):

    directory = "../Plot/"
    if num == "0":
        file = "AVG" + type + num + ".png"
    else:
        file = type + num + ".png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + file)

def fromCSVToLatexTable(nome1, nome2):
    df = pd.read_csv("CSVresult/"+nome1+".csv")
    a = df.values
    a = a[:, 1:]
    np.savetxt("CSVresult/Latextable"+nome2+".csv", a, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')
