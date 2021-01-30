
import matplotlib.pyplot as plt
import seaborn as sns

import random
import math
import numpy as np
from scipy import spatial

def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)

    
def kmeans(num_of_clusters,dataset, dimension):

    Max_Iterations = 1000
    i = 0
    
    cluster = [0] * number_of_samples
    prev_cluster = [-1] * number_of_samples
    
    cluster_centers = []
    for i in range(0,num_of_clusters):
        new_cluster = []
        cluster_centers += [random.choice(dataset)]
        
        force_recalculation = False
    
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        for p in range(0,number_of_samples):
            min_dist = float("inf")
            
            for c in range(0,len(cluster_centers)):
                
                dist = eucldist(dataset[p],cluster_centers[c])
                
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c 
        
        
        for k in range(0,len(cluster_centers)):
            new_center = [0] * dimension
            members = 0
            for p in range(0,number_of_samples):
                if (cluster[p] == k): 
                    for j in range(0,dimension):
                        new_center[j] += dataset[p][j]
                    members += 1
            
            for j in range(0,dimension):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                
                else: 
                    new_center = random.choice(dataset)
                    force_recalculation = True
                    
            
            cluster_centers[k] = new_center
    
    pts = np.array(dataset)
    candidates = pts[spatial.ConvexHull(pts).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    
    colors = ['b','g','r','y','c','k','m']


    for cluster_index in range(0,num_of_clusters):
        samples = []
        for i in range(0,number_of_samples):
            if cluster[i] == cluster_index:
                samples += [dataset[i]]
                sample = dataset[i]
                plt.scatter(x = sample[0], y = sample[1], color = colors[cluster_index])
        
        pts = np.array(samples)
        candidates = pts[spatial.ConvexHull(pts).vertices]

        dist_mat = spatial.distance_matrix(candidates, candidates)

        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        print(candidates[i], candidates[j])
        print("MAX DISTANCE")
        print(eucldist(candidates[i], candidates[j]))
    

        centr = cluster_centers[cluster_index]

        print("CENTROID: ",centr[0],centr[1])

        plt.scatter(x = centr[0], y = centr[1], color = 'black', marker = 'x')
        plt.title("K-means")

    plt.show()



def read_input(filename):
    input = open(filename, "r")
    number_of_samples, dimension = [int(i) for i in input.readline().split()]
    dataset = [[float(j) for j in input.readline().split()] for i in range(number_of_samples)] 
    return number_of_samples, dimension, dataset




if __name__ == "__main__":
    
    number_of_samples, dimension, dataset = read_input("g2-2-100.txt")

    print(number_of_samples, dimension)
    
    k = 3

    kmeans(k,dataset, dimension) 