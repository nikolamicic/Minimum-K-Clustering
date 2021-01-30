from particle import Particle


import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import math

from kmeans import KMeans

def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)


class ParticleSwarmOptimizedClustering:
    def __init__(self,
                 n_cluster: int,
                 n_particles: int,
                 data: np.ndarray,
                 hybrid: bool = True,
                 max_iter: int = 100,
                 print_debug: int = 10):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid

        self.print_debug = print_debug
        self.gbest_score = np.inf
        self.gbest_centroids = None
        self.gbest_cluster = None
        self._init_particles()

    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            if i == 0 and self.hybrid:
                particle = Particle(self.n_cluster, self.data, use_kmeans=True)
            else:
                particle = Particle(self.n_cluster, self.data, use_kmeans=False)
            if particle.best_score < self.gbest_score:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_score = particle.best_score
                self.gbest_cluster = particle.cluster.copy()
            self.particles.append(particle)

    def run(self):
        print('Initial global best score', self.gbest_score)
        history = []
    
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.gbest_cluster,self.gbest_centroids, self.data)
            for particle in self.particles:
                if particle.best_score < self.gbest_score:
                    self.gbest_centroids = particle.centroids.copy()
                    self.gbest_score = particle.best_score
                    self.gbest_cluster = particle.cluster.copy()        
            history.append(self.gbest_score)
            if i % self.print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                    i + 1, self.max_iter, self.gbest_score))

                
        print('Finish with gbest score {:.18f}'.format(self.gbest_score))
        colors = ['b','g','r','c','m','y','k']
        for cluster_index in range(0,self.n_cluster):
            samples = []
            for i in range(0,500):
                if self.gbest_cluster[i] == cluster_index:
                    samples += [self.data[i]]
                    sample = self.data[i]

                    plt.scatter(x = sample[0], y = sample[1], color = colors[cluster_index])      
            pts = np.array(samples)
            candidates = pts[spatial.ConvexHull(pts).vertices]
            dist_mat = spatial.distance_matrix(candidates, candidates)
            i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

            print(candidates[i], candidates[j])
            print("MAX DISTANCE")
            print(eucldist(candidates[i], candidates[j]))
            

            centr = self.gbest_centroids[cluster_index]

            # #     print("CENTROID: ",centr[0],centr[1])

            plt.scatter(x = centr[0], y = centr[1], color = 'black', marker = 'x')
            plt.title("PSO")

        plt.show()
        return history


if __name__ == "__main__":
    pass