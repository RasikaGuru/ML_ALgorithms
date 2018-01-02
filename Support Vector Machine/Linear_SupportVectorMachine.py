#Authors: Kavya Sethuram, Rasika Guru, Roshani Mangalore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
import pylab as pl
if __name__=="__main__":
    data=pd.read_csv("linsep.txt",sep=',',header=None)
    dataarray=np.array(data)
    classifier = np.array(dataarray)[:, 2]
    X = np.array(dataarray)[:,0:2]
    classifier = np.resize(classifier, (100, 1))
    P = matrix(np.dot(X, X.T) * np.dot(classifier, classifier.T))
    q = matrix(np.ones(100) * -1)
    G = matrix(np.diag(np.ones(100) * -1))
    h = matrix(np.zeros(100))
    b = matrix([0.0])
    A = matrix(classifier.T, (1, 100))
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = sol['x']
    weight = 0.0
    for i in range(100):
        weight += alpha[i] * classifier[i] * X[i]
    print ('weight=', weight)
    fcoord = []
    falpha = []
    fval = []
    for i in range(100):
        if (alpha[i] > 0.0001):
            falpha.append(alpha[i])
            fcoord.append(X[i])
            fval.append(classifier[i])
    print ("alpha=", falpha)
    b = (1 / fval[0]) - np.dot(weight, fcoord[0])
    print ('b=',b)

    bias = b[0]

    # normalize
    norm = np.linalg.norm(weight)
    weight, bias = weight / norm, bias / norm




