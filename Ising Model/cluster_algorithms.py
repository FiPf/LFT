from base_simulation import Base_sim
from ising_model_physics import IsingModel
import numpy as np
from abc import ABC

class ClusterAlgorithm(IsingModel, ABC): 
    def __init__(self, shape, B, J, beta):
        super().__init__(shape, B, J, beta)

    def generate_clusters(self): 
        cluster_number = 0
        clusters = np.full_like(self.lattice, -1)
        cluster_sizes = []
        for i in np.ndindex(clusters.shape): 
            if clusters[*i] == -1: 
                self.generate_cluster(i, clusters, cluster_number)
                cluster_number += 1
                cluster_sizes.append(self.cluster_size)
        self.cluster_sizes = cluster_sizes
        return (clusters == i for i in range(cluster_number))
    
    def generate_cluster(self, start_index, clusters=None, cluster_nr=0):
        if clusters is None:
            clusters = np.full_like(self.lattice, -1)

        start_spin = self.lattice[*start_index]
        clusters[*start_index] = cluster_nr
        cluster_size = 1

        # Initialize the cluster mask with the starting spin
        cluster_mask = np.zeros_like(self.lattice, dtype=bool)
        cluster_mask[*start_index] = True

        added = True
        while added:
            added = False
            # Iterate over all spins currently in the cluster
            for index, val in np.ndenumerate(self.lattice):
                if cluster_mask[*index]:
                    for n in self.neighbours(index):
                        if not cluster_mask[*n] and self.lattice[*n] == start_spin:
                            if np.random.rand() > np.exp(-2 * self.beta * self.J):
                                cluster_mask[*n] = True
                                clusters[*n] = cluster_nr
                                cluster_size += 1
                                added = True

        self.cluster_size = cluster_size
        return clusters == cluster_nr
    
class SwendsenWang(ClusterAlgorithm): 
    def __init__(self, shape, B, J, beta):
        super().__init__(shape, B, J, beta)

    def update(self): 
        super().update()

        clusters = self.generate_clusters()
        for cluster in clusters: 
            if np.random.rand() < 0.5: 
                self.lattice[cluster] *= -1

    def get_susceptibility(self):
        return sum(cluster_size**2 for cluster_size in self.cluster_sizes) / np.size(self.lattice)

class Wolff(ClusterAlgorithm): 
    def __init__(self, shape, B, J, beta):
        super().__init__(shape, B, J, beta)

    def update(self): 
        super().update()
        index = np.random.randint(self.get_lattice().shape)
        self.lattice[self.generate_cluster(index)] *= -1

    def get_susceptibility(self):
        return self.cluster_size