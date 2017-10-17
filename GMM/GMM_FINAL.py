import itertools
import pandas as pd
import numpy as np
import math
import copy
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg

## Function that calculates mean and amplitude

def calc_mean_amplitude(random_weights):
    mean_list = []
    amplitude_list = []
    for i in range(0,3,1):
        sum_weights = 0.0
        mean = [0.0, 0.0]
        for j in range(0,len(data_array),1):
            mean = mean+ data_array[j]*random_weights[j][i]
            sum_weights = sum_weights+random_weights[j][i]
        mean_list.append(mean/sum_weights)
        amplitude_list.append(sum_weights/150.0)
    return mean_list,amplitude_list

## Function that calculates mean matrices (X-mean)
def calculate_meanmatrices(mean_list):
    mean_matrices = []
    for i in range(0,3,1):
        data_mean_array = df.subtract(mean_list[i]).as_matrix()
        mean_matrices.append(data_mean_array)
    return mean_matrices


## Function that calculates covariant matrices

def covariant(mean_matrices):
    covariant_list = []
    for i in range(len(mean_matrices)):
        covariant_matrix = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        weights = 0.0
        for j in range(0,150,1):
            covariant_matrix = covariant_matrix + (random_weights[j][i] * np.matrix(mean_matrices[i][j]).T * np.matrix(mean_matrices[i][j]))
            weights = weights+random_weights[j][i]
        covariant_list.append(covariant_matrix/weights)

    return covariant_list


## Find Gaussian PDF

def guassian(x,mean,covariantmat):
    det=np.linalg.det(covariantmat)
    inv=np.linalg.inv(covariantmat)
    epower=math.exp((-0.5)*np.matrix(x)*np.matrix(inv)*np.matrix(x).T)
    return (((det)**(-0.5))*(epower))/(2*math.pi)

## Finding weights

def individual_weights(random_weights):

    mean_list,amplitude_list = calc_mean_amplitude(random_weights)
    mean_matrices = calculate_meanmatrices(mean_list)
    covariant_list = covariant(mean_matrices)
    act_weight_list = []
    convergence_list = []

    for i in range(150):

        weight_list = []
        sum_weight = 0.0
        for c in range(3):
            weight_list.append(amplitude_list[c]*guassian(mean_matrices[c][i],mean_list[c],covariant_list[c]))
            sum_weight = sum_weight + weight_list[c]
        #convergence_list.append(weight_list)

        weight_list = weight_list/sum_weight
        act_weight_list.append(weight_list)


    return mean_list,amplitude_list,covariant_list,act_weight_list


def convergence(convergence_list):
    total = 0
    for i in range(150):
        total = total + math.log(convergence_list[i][0]+convergence_list[i][1]+convergence_list[i][2])
    return total



def plot_guassians(X,means, covariances, index, title):
    # from pprint import pprint
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, c, color) in enumerate(zip(
            means, covariances, color_iter)):

        # pprint(c)
        v, w = linalg.eigh(c)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.3)
    splot.add_artist(ell)
    plt.xlim(-10., 10.)
    plt.ylim(-10., 10.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], color="R")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('clusters.txt', header=None)
    data_array = df.as_matrix()
    result = []
    random_weights = []
    count = 0
    current_conv = 0
    prev_conv = 0
    global color_iter
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue'])
    for i in range(0, 150, 1):
        r1 = random.uniform(0.1,0.4)
        r2 = random.uniform(0.1,0.4)
        r3 = 1 - (r1 + r2)
        random_weights.append([r1, r2, r3])

    while (True):
        result = individual_weights(random_weights)
        print result [0]
        random_weights = result[3]
        prev_conv = current_conv
        current_conv = convergence(result[3])
        print count
        if count!=0:
            if current_conv >= prev_conv:
                print prev_conv
                print current_conv
                break
        count = count+1

    # plot_guassians(data_array, np.array(result[0]),np.array(result[2]), 1, 'Result')

    #result[0]=[list(x) for x in result[0]]
    # print result[0]
    # print result[1]
    # print result[2]


    # plt.scatter(result[0][0][0],result[0][0][1],color="R")
    # plt.scatter(result[0][1][0], result[0][1][1], color="Y")
    # plt.scatter(result[0][2][0], result[0][2][1], color="C")
    # plt.scatter(data_array[:, 0], data_array[:, 1])
    # plt.show()