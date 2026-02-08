import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

class TopologyAwareFeatureProjection:
    def __init__(self, edges, feature_length=10, order=3):
        self.edges = edges
        self.feature_length = feature_length
        self.order = order
        self.nodes = self._get_nodes()
        self.adjacency_matrix = self._build_adjacency_matrix()
        #print("Adjacency matrix built successfully.")
        self.projection = self._compute_topology_aware_projection()
        #print("Topology-aware projection computed.")

    def _get_nodes(self):
        nodes = set()
        for src, dst in self.edges:
            nodes.add(src)
            nodes.add(dst)
        return sorted(list(nodes))

    def _build_adjacency_matrix(self):
        num_nodes = len(self.nodes)
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes))
        for src, dst in self.edges:
            src_index = self.nodes.index(src)
            dst_index = self.nodes.index(dst)
            adjacency_matrix[src_index, dst_index] = 1
            adjacency_matrix[dst_index, src_index] = 1
        return adjacency_matrix.tocsr()

    def _compute_topology_aware_projection(self):
        adjacency_matrix = self.adjacency_matrix.astype(float)
        degrees = adjacency_matrix.sum(axis=1).A.flatten()
        d_inv_sqrt = sp.diags(np.power(degrees, -0.5))
        normalized_adjacency_matrix = d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt
        smoothed_adjacency_matrix = sum(np.linalg.matrix_power(normalized_adjacency_matrix.toarray(), i)
                                        for i in range(1, self.order + 1))

        #SVD
        u, s, vt = svds(smoothed_adjacency_matrix, k=self.feature_length)
        sigma = np.diag(s)
        projected_features = u @ sigma
        projected_features = (projected_features - np.mean(projected_features, axis=0)) / \
                             np.std(projected_features, axis=0)

        #Mapping
        features_mapping = {self.nodes[i]: projected_features[i, :]
                            for i in range(len(self.nodes))}

        return features_mapping

    def get_node_features(self):
        return self.projection
