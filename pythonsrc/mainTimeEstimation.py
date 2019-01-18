import numpy as np
import conjugateGradient as CG
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from utility import *
import csv

espereiments = 100
numberOfMatrix = 10
matrixes = ["A", "B","C", "D", "E", "F", "G", "H"]

def main():
    writer = csv.writer(open("CSVresult/MAtrixStat.csv", 'w'))
    writer.writerow(["MAtrixName", "Time mean Numpy linalog", "Time std Numpy linalog",
     "Time mean steepestDescent" ,  "Time std steepestDescent", 
     " mean log10 error steepest descent", " std log10 error steepest descent",
    "mean Time Conjugate gardient","std Time Conjugate gardient",
     "mean log10 error Conjugate gardient", "std log10 error Conjugate gardient"])
    for typeMatrix in matrixes:
        print("matrix type "+typeMatrix)
        for i in range(1,numberOfMatrix+1):
            A = readMatrix(typeMatrix,i)
            CGtimes = []
            SGDtimes = []
            numpytimes = []
            errorSGD = []
            errorCG = []
            for i in range(espereiments):
                # numpy norm
                startime = time.time()
                normDEF = LA.norm(A, ord = 2)**2
                numpytimes.append(time.time()- startime)
                # conjugate gradient
                startime = time.time()
                _, norm = normCG(A)
                CGtimes.append(time.time()- startime)
                errorCG.append(abs(norm - normDEF)/normDEF)
                # steepest gradient descent
                startime = time.time()
                _, norm = normSDG(A)
                SGDtimes.append(time.time()- startime)
                errorSGD.append(abs(norm - normDEF)/normDEF)       
        CGtimes = np.array(CGtimes)
        SGDtimes = np.array(SGDtimes)
        numpytimes = np.array(numpytimes)
        errorCG = [max(err, 1e-20) for err in errorCG]
        errorSGD = [max(err, 1e-20) for err in errorSGD]
        errorSGD = np.array(errorSGD)
        errorCG = np.array(errorCG)
        print("Conjugate gradient time %.6f" %(np.mean(CGtimes)))
        print("Steepest gradient descent average time %.6f" %(np.mean(SGDtimes)))
        print(" Numpy ninealog average time %.6f" %(np.mean(numpytimes)))
        print("Error conjugate gradient = %.6f" % (np.mean(np.log10(errorCG))))
        print("Error steppest descent direct6ion = %.6f" %np.mean(np.log10(errorSGD)))
        writer.writerow([typeMatrix, np.mean(numpytimes), np.std(numpytimes),
        np.mean(SGDtimes), np.std(SGDtimes), np.mean(np.log10(errorSGD)),np.mean(np.log10(errorSGD)),
        np.mean(CGtimes), np.std(CGtimes), np.mean(np.log10(errorCG)),np.mean(np.log10(errorCG))])

if __name__ == "__main__":
    main()