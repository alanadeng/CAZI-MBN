class LayerProcessor:
    def __init__(self, nodes, train_edges, val_edges, test_edges, features_dict, num_layers = 6, weights = None):
        self.nodes = nodes
        self.num_layers = num_layers
        self.weights = weights # layer weights
        self.features_dict = features_dict  # Store the original features dictionary
        self.original_edges = {
            'train_edges': train_edges,
            'val_edges': val_edges,
            'test_edges': test_edges,
        }
        self.layered_edges = {key: [[] for _ in range(num_layers)] for key in
                              self.original_edges.keys()}  # Prepare lists for layers
        self.node_layer_map = self._create_node_layer_map()
        self.layered_features = [[] for _ in range(num_layers)]  # Prepare lists for layers
        self.id_map = []
        self._process_edges()

    def _create_node_layer_map(self):
        """Create a mapping from node ID to its layer."""
        return {node['id']: node['layer'] for node in self.nodes}

    def _reindex_nodes_in_layer(self, layer_nodes):
        """Re-index nodes in a layer, starting from 0."""
        return {original_id: new_id for new_id, original_id in enumerate(sorted(layer_nodes))}

    def _process_edges(self):
        """Sort edges into layers, re-index node IDs based on layer position, and extract features."""
        for layer in range(0, self.num_layers):
            layer_nodes = set()
            for edge_type, edges in self.original_edges.items():
                for src_id, dst_id in edges:
                    if self.node_layer_map[src_id] == layer:
                        layer_nodes.update([src_id, dst_id])

            # Re-index nodes for the current layer
            reindex_map = self._reindex_nodes_in_layer(layer_nodes)
            self.id_map.append(reindex_map)

            # Process features for re-indexed nodes
            self.layered_features[layer - 1] = {reindex_map[node_id]: self.features_dict[node_id] for node_id in
                                                layer_nodes if node_id in self.features_dict}


            # Re-index edges for the current layer
            for edge_type, edges in self.original_edges.items():
                self.layered_edges[edge_type][layer - 1] += [
                    (reindex_map[src_id], reindex_map[dst_id]) for src_id, dst_id in edges
                    if self.node_layer_map[src_id] == layer and src_id in reindex_map and dst_id in reindex_map
                ]

    def get_layered_data(self):
        """Get the processed, layered edges and features."""
        return self.layered_edges, self.layered_features

    def create_layer_datasets(self):
        datasets = []
        layered_edges, layered_features = self.get_layered_data()
        # Create datasets by layer
        for layer in range(self.num_layers-1, -1, -1):
            train_edges = layered_edges['train_edges'][layer]
            val_edges = layered_edges['val_edges'][layer]
            test_edges = layered_edges['test_edges'][layer]
            features_dict = layered_features[layer]

            dataset = (features_dict, train_edges, val_edges, test_edges)
            #dataset = (features_dict, train_edges, val_edges)
            datasets.append(dataset)
        if self.weights == None:
            datasets.sort(key=lambda x: len(x[0]))
            self.id_map.sort(key=lambda x: len(x))
        else:
            datasets.sort(key=self.weights)
            self.id_map.sort(key=self.weights)
        return datasets, self.id_map
