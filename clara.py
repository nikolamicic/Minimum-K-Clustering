import pandas as pd
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import copy
from scipy import spatial

def distance(x, y):
    d = 0
    for i in range(len(x)):
        d += (x[i]-y[i])**2
    d = np.sqrt(d)
    return d

def cluster(X, M, k):
    C = []
    cost = 0
    for i in range(k):
        C.append([])
    for i in range(len(X)):
        minidx = 0
        mindist = np.inf
        for j in range(k):
            d = distance(X[i],M[j])
            if(d < mindist):
                minidx = j
                mindist = d
        C[minidx].append(i)
        cost += mindist
    return C, cost

def calMedoids(X, C, M, k):
    newMed = rn.choice(X)
    N = copy.deepcopy(M)
    if(newMed not in M):
        for j in range(k):
            N = copy.deepcopy(M) 
            N[j] = newMed
            CM, m = cluster(X, M, k)
            CN, n = cluster(X, N, k)
            if(n < m):
                break
    return N

def initialSeed(X, k):
    Sel = []
    n = len(X)
    a = rn.choice(X)
    Sel.append(a)
    dis = np.zeros(n)
    for i in range(k-1):
        for i in range(n):
            dis[i] += distance(X[i], a)
        m = np.argmax(dis)
        a = X[m]
        Sel.append(a)
    Sel = np.array(Sel)
    return Sel

def clara(X, k, TS, SS, iter = 100):
    M = initialSeed(X, k)
    bestCost = np.inf
    for i in range(TS):
        S = rn.choices(X, k=SS)
        for lp in range(iter):
            C, cost = cluster(S, M, k)
            N = calMedoids(S, C, M, k)
            M = copy.deepcopy(N)
        if(cost < bestCost):
            bestM = copy.deepcopy(M)
            bestCost = cost
    C, cost = cluster(X, bestM, k)
    return C



def read_input(filename):
    input = open(filename, "r")
    number_of_samples, dimension = [int(i) for i in input.readline().split()]
    dataset = [[float(j) for j in input.readline().split()] for i in range(number_of_samples)] 
    return number_of_samples, dimension, dataset


if __name__ == '__main__':
    n_samples,n_features,X = read_input("b2-sub-10.txt")
    X = np.array(X)
    print(X)
    
    res = clara(X, 3, 20, 200)

    color = ['b','g','r','y','c','k','m']
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in range(len(res)):
        lst = []
        col = color[i]
        for j in range(len(res[i])):
            lst.append(X[res[i][j]])
        pts = np.array(lst)
        candidates = pts[spatial.ConvexHull(pts).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
        c_i, c_j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        print(candidates[c_i], candidates[c_j])
        print("MAX DISTANCE FOR CLUSTER ", i)
        print(distance(candidates[c_i], candidates[c_j]))


        lst = pd.DataFrame(lst)
        ax.scatter(lst[0],lst[1],c=col)
    plt.show()
