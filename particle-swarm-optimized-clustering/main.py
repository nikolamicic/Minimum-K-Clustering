import numpy as np
import pandas as pd

from pso import ParticleSwarmOptimizedClustering
from utils import normalize
def read_input(filename):
    input = open(filename, "r")
    number_of_samples, dimension = [int(i) for i in input.readline().split()]
    dataset = [[float(j) for j in input.readline().split()] for i in range(number_of_samples)] 
    return dataset

if __name__ == "__main__":
    data = read_input("g2-2-100.txt")
    x = pd.DataFrame(data)
    print(len(x))
    x = x.values
    print(x)
    pso = ParticleSwarmOptimizedClustering(
        n_cluster=3, n_particles=40, data=x, hybrid=True, #max_iter=2000, print_debug=50
        )
    pso.run()