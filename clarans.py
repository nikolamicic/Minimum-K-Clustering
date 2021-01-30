import math
import random
import numpy as np
import sys
from past.builtins import xrange

import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy import spatial

def clarans_basic(points, numlocal, maxneighbor, mincost,k):
    i=1
    N = len(points)
    d_mat = np.asmatrix(np.empty((k,N)))
    local_best = []
    bestnode = []
    
    while i<=numlocal:
        node = np.random.permutation(range(N))[:k]
        fill_distances(d_mat, points, node)     
        cls = assign_to_closest(points, node, d_mat)   
        cost = total_dist(d_mat, cls)
        copy_node = node.copy()
        print ('new start \n')
        j = 1 
        
        while j<=maxneighbor:
            changing_node = copy_node.copy()
            idx = pick_random_neighbor(copy_node, N)
            update_distances(d_mat, points, copy_node, idx)            
            cls = assign_to_closest(points, copy_node, d_mat)   
            new_cost = total_dist(d_mat, cls)
            
            if new_cost < cost:
                cost = new_cost
                local_best = copy_node.copy()
                print ('Best cost: ' + str(cost) + ' ')
                print (local_best)
                print ('\n')
                j = 1
                continue
            else:
                j=j+1
                if j<=maxneighbor:
                    continue
                elif j>maxneighbor:
                    if mincost>cost:
                        mincost = cost
                        print ("change bestnode ")
                        print (bestnode)
                        print (" into")
                        bestnode = local_best.copy()
                        print (bestnode)
                        print ('\n')
                        
            i = i+1
            if i>numlocal:
                fill_distances(d_mat, points, bestnode)     
                cls = assign_to_closest(points, bestnode, d_mat)
                print ("Final cost: " + str(mincost) + ' ')
                print (bestnode)
                print ('\n')
                return cls, bestnode
            else:
                break
    
    
def pick_random_neighbor(current_node, set_size):
    node = random.randrange(0, set_size, 1)
    while node in current_node:
        node = random.randrange(0, set_size, 1)
        
    i = random.randrange(0, len(current_node))
    current_node[i]=node
    return i
    
def dist_euc(t1, t2):
    return math.sqrt((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2)

def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)

def assign_to_closest(points, meds, d_mat):
    cluster =[]
    for i in xrange(len(points)):
        if i in meds:
            cluster.append(np.where(meds==i))
            continue
        d = sys.maxsize
        idx=i
        for j in xrange(len(meds)):
            d_tmp = d_mat[j,i]
            if d_tmp < d:
                d = d_tmp
                idx=j
        cluster.append(idx)
    return cluster


def fill_distances(d_mat, points, current_node):
    for i in range(len(points)):
        for k in range(len(current_node)):
            d_mat[k,i]=eucldist(points[current_node[k]], points[i])
        
        
def total_dist(d_mat, cls):
    tot_dist = 0
    for i in xrange(len(cls)):
        tot_dist += d_mat[cls[i],i]
    return tot_dist


def update_distances(d_mat, points, node, idx):
    for j in range(len(points)):
        d_mat[idx,j]=eucldist(points[node[idx]], points[j])

def read_input(filename):
    input = open(filename, "r")
    n_samples, dimension = [int(i) for i in input.readline().split()]
    dataset = [[float(j) for j in input.readline().split()] for i in range(n_samples)] 
    return n_samples, dimension, dataset


if __name__ == '__main__':
    n_samples,n_features,X = read_input("b2-sub-10.txt")
    X = np.array(X)
    print(X)
    num_of_clusters = 3
    res1,res2 = clarans_basic(X, 7, 100, 99000000,num_of_clusters)
    #mincost je ovoliko veliki zbog velikog skupa podataka
    print("Res1", res1)
    print("Res2", res2)

    colors = ['b','g','r','y','c','k','m']

    for cluster_index in range(0,num_of_clusters):
        samples = []
        for i in range(0,n_samples):
            if res1[i] == cluster_index:
                samples += [X[i]]
                sample = X[i]

                plt.scatter(x = sample[0], y = sample[1], color = colors[cluster_index])
        
        pts = np.array(samples)
        candidates = pts[spatial.ConvexHull(pts).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)

        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        print(candidates[i], candidates[j])
        print("MAX DISTANCE")
        print(eucldist(candidates[i], candidates[j]))
        centr = X[res2[cluster_index]]

        plt.scatter(x = centr[0], y = centr[1], color = 'black', marker = 'x')


    plt.show()
