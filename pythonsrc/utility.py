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
              A=None, type=None, num=None):
    yLabel1 = 'Absolute Error'
    yLabel2 = 'Relative Error'
    yLabel3 = 'Norms Gradient'
    xLabel1 = 'Iterations'

    fig, [relErrPlot, gradPlot] = plt.subplots(2, 2, sharex=False, sharey=True)
    fig.set_size_inches(18.5, 10.5)
    m, n = np.shape(A)

    relErrPlot[0].set_title(
        'Steepest Gradient Descent \n Type ' + type + '     Density =  ' + str(density(type)) + '    M = ' + str(
            m) + ' N = ' + str(n))
    # errPlot[0].set(ylabel=yLabel1)
    # errPlot[0].set_yscale('log')
    # errPlot[0].plot(errorsSGD)

    relErrPlot[0].set_yscale('log')
    relErrPlot[0].set(ylabel=yLabel2)
    for relSGD in relerrorsSGD:
        relErrPlot[0].plot(relSGD)

    gradPlot[0].set(ylabel=yLabel3)
    gradPlot[0].set(xlabel=xLabel1)
    gradPlot[0].set_yscale('log')

    for gradSGD in gradientsSGD:
        gradPlot[0].plot(gradSGD)

    relErrPlot[1].set_title(
        'Conjugate Gradient \n Type ' + type + '     Density =  ' + str(density(type)) + '    M = ' + str(
            m) + ' N = ' + str(n))
    # errPlot[1].set_yscale('log')
    # errPlot[1].plot(errorsCG, "C1")

    relErrPlot[1].set_yscale('log')
    for relCG in relerrorsCG:
        relErrPlot[1].plot(relCG)

    gradPlot[1].set(xlabel=xLabel1)
    gradPlot[1].set_yscale('log')
    for gradCG in gradientsCG:
        gradPlot[1].plot(gradCG)

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

def fromCSVToLatexTable():
    df = pd.read_csv("CSVresult/MAtrixStat.csv")
    a = df.values
    a = a[:, 1:]
    np.savetxt("CSVresult/mydata.csv", a, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')
