"""
Phase 2, Milestone M2.1 — Online Clustering for Concept Formation
Uses a Self-Organizing Map (SOM) to cluster perception vectors into proto-concepts.
"""

import numpy as np


class SelfOrganizingMap:
    def __init__(self, m=10, n=10, dim=256, learning_rate=0.5, sigma=None):
        self.m = m
        self.n = n
        self.dim = dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma else max(m, n) / 2
        self.weights = np.random.rand(m, n, dim)

    def _find_bmu(self, x):
        """Find Best Matching Unit (BMU) for input vector x."""
        diff = self.weights - x.reshape(1, 1, self.dim)
        dist = np.linalg.norm(diff, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dist), (self.m, self.n))
        return bmu_idx

    def _update_weights(self, x, bmu_idx, iteration, max_iterations):
        """Update SOM weights based on BMU and neighborhood function."""
        lr = self.learning_rate * np.exp(-iteration / max_iterations)
        sigma = self.sigma * np.exp(-iteration / max_iterations)

        for i in range(self.m):
            for j in range(self.n):
                dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                if dist_to_bmu <= sigma:
                    influence = np.exp(-(dist_to_bmu ** 2) / (2 * sigma ** 2))
                    self.weights[i, j, :] += influence * lr * (x - self.weights[i, j, :])

    def train(self, data, num_iterations=1000):
        for iteration in range(num_iterations):
            x = data[np.random.randint(0, len(data))]
            bmu_idx = self._find_bmu(x)
            self._update_weights(x, bmu_idx, iteration, num_iterations)

    def map_vector(self, x):
        """Map a vector to its BMU index (concept ID)."""
        return self._find_bmu(x)


if __name__ == "__main__":
    som = SelfOrganizingMap(m=5, n=5, dim=256)
    dummy_data = np.random.rand(100, 256)
    som.train(dummy_data, num_iterations=10)
    concept_id = som.map_vector(np.random.rand(256))
    print("Mapped to concept ID:", concept_id)
