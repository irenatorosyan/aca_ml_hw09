import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph


class SpectralClustering:
    def __init__(self, n_clusters=2, n_neighbors=10):
        self.n_clusters = n_clusters
        self.affinity = 'nearest_neighbors'
        self.n_neighbors = n_neighbors
        self.eigenvalues = None
        self.eigenvectors = None
        self.similarity = None
        self.laplacian = None
        self.selected_eigenvectors = None
        self.labels = None

    def fit(self, X):
        adjacency = kneighbors_graph(X, n_neighbors=self.n_neighbors, metric='euclidean', include_self=True)
        adjacency = 0.5 * (adjacency + adjacency.T)
        self.similarity = adjacency.toarray()

        degree = np.diag(np.sum(self.similarity, axis=1))
        self.laplacian = degree - self.similarity

        eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian)
        indices = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[indices]
        self.eigenvectors = eigenvectors[:, indices]

        selected_eigenvectors = self.eigenvectors[:, :self.n_clusters]
        self.selected_eigenvectors = normalize(selected_eigenvectors)

        self.labels = KMeans(n_clusters=self.n_clusters).fit_predict(eigenvectors)

    def predict(self, X):
        return KMeans(n_clusters=self.n_clusters).fit_predict(self.selected_eigenvectors)
