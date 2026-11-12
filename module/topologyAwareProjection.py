import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

class TopologyAwareFeatureProjection:
    def __init__(self, edges, feature_length=10, order=3, nodes=None):
        self.edges = edges
        self.feature_length = feature_length
        self.order = order
        self.nodes = self._get_nodes(nodes)
        self.adjacency_matrix = self._build_adjacency_matrix()
        self.projection = self._compute_topology_aware_projection()

    def _get_nodes(self, nodes=None):
        node_set = set(nodes) if nodes is not None else set()
        for src, dst in self.edges:
            node_set.add(src)
            node_set.add(dst)
        return sorted(node_set)

    def _build_adjacency_matrix(self):
        num_nodes = len(self.nodes)
        node_indices = {node: index for index, node in enumerate(self.nodes)}
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for src, dst in self.edges:
            src_index = node_indices[src]
            dst_index = node_indices[dst]
            adjacency_matrix[src_index, dst_index] = 1
            adjacency_matrix[dst_index, src_index] = 1
        return adjacency_matrix.tocsr()

    def _compute_topology_aware_projection(self):
        if self.feature_length <= 0:
            raise ValueError("feature_length must be greater than zero.")
        if self.order <= 0:
            raise ValueError("order must be greater than zero.")

        total_nodes = len(self.nodes)
        if total_nodes == 0:
            return {}

        full_degrees = self.adjacency_matrix.sum(axis=1).A.flatten()
        active = np.flatnonzero(full_degrees > 0)
        if active.size == 0:
            return {node: np.zeros(self.feature_length, dtype=np.float32) for node in self.nodes}
        num_nodes = active.size
        adjacency_matrix = self.adjacency_matrix[active][:, active].astype(np.float32)
        degrees = adjacency_matrix.sum(axis=1).A.flatten()
        inverse_sqrt_degrees = np.zeros_like(degrees)
        nonzero_degrees = degrees > 0
        inverse_sqrt_degrees[nonzero_degrees] = degrees[nonzero_degrees] ** -0.5
        d_inv_sqrt = sp.diags(inverse_sqrt_degrees)
        normalized_adjacency_matrix = d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt
        smoothed_adjacency_matrix = normalized_adjacency_matrix.copy()
        current_power = normalized_adjacency_matrix.copy()
        for _ in range(1, self.order):
            current_power = current_power @ normalized_adjacency_matrix
            smoothed_adjacency_matrix = smoothed_adjacency_matrix + current_power

        # Split the requested width across both SVD factors; an odd extra channel is zero-padded.
        component_count = min(max(1, self.feature_length // 2), num_nodes)
        if component_count <= 0 or smoothed_adjacency_matrix.nnz == 0:
            projected_features = np.zeros((num_nodes, self.feature_length), dtype=np.float32)
        else:
            if component_count < num_nodes:
                u, singular_values, vt = svds(
                    smoothed_adjacency_matrix,
                    k=component_count,
                    v0=np.ones(min(smoothed_adjacency_matrix.shape), dtype=np.float32),
                )
            else:
                u, singular_values, vt = np.linalg.svd(
                    smoothed_adjacency_matrix.toarray(),
                    full_matrices=False,
                )
                u = u[:, :component_count]
                singular_values = singular_values[:component_count]
                vt = vt[:component_count, :]

            order = np.argsort(singular_values)[::-1]
            singular_values = singular_values[order]
            u = u[:, order]
            vt = vt[order, :]
            projected_features = np.zeros((num_nodes, self.feature_length), dtype=np.float32)
            max_singular_value = (
                singular_values.max() if singular_values.size else 0.0
            )
            tolerance = np.finfo(np.float32).eps * max_singular_value
            rank = int(np.count_nonzero(singular_values > tolerance))
            scale = np.sqrt(singular_values[:rank])
            left = u[:, :rank] * scale
            right = vt[:rank, :].T * scale
            factors = np.concatenate((left, right), axis=1)
            copied_width = min(self.feature_length, factors.shape[1])
            projected_features[:, :copied_width] = factors[:, :copied_width]
            projected_features[degrees == 0] = 0
            mean = projected_features.mean(axis=1, keepdims=True)
            variance = projected_features.var(axis=1, keepdims=True)
            projected_features = (
                (projected_features - mean)
                / np.sqrt(variance + 1e-5)
            ).astype(np.float32)
            projected_features[degrees == 0] = 0

        full_features = np.zeros((total_nodes, self.feature_length), dtype=np.float32)
        full_features[active] = projected_features
        features_mapping = {self.nodes[i]: full_features[i, :]
                            for i in range(total_nodes)}

        return features_mapping

    def get_node_features(self):
        return self.projection
