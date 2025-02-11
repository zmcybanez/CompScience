def euclidean_distance(x1, x2):
    return sum((x1_i - x2_i) ** 2 for x1_i, x2_i in zip(x1, x2)) ** 0.5

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples = len(X)
        self.n_features = len(X[0])
        random_sample_idxs = [i for i in range(self.n_samples)][:self.K]
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(centroids_old, self.centroids):
                break
        
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = [0] * self.n_samples
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        return distances.index(min(distances))

    def _get_centroids(self, clusters):
        centroids = [[0] * self.n_features for _ in range(self.K)]
        for cluster_idx, cluster in enumerate(clusters):
            cluster_points = [self.X[i] for i in cluster]
            centroids[cluster_idx] = [sum(dim) / len(cluster) for dim in zip(*cluster_points)]
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

if __name__ == "__main__":
    X = [[i, j] for i in range(10) for j in range(10)]
    clusters = 3
    k = KMeans(K=clusters, max_iters=10)
    y_pred = k.predict(X)
    print("Predictions:", y_pred)
