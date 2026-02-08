class EdgeExtractor:
    def __init__(self, graph_json, seed=0):
        self.graph_json = graph_json
        self.edges, self.set_type_edges = self._extract_edges()


    def _extract_edges(self):
        edges = []
        set_type_edges = {'train': [], 'val': [], 'test': []}
        for edge in self.graph_json['edges']:
            pair = (edge['source'], edge['target'])
            edges.append(pair)
            # Treat inter-layer edges as part of training set and intra-layer edges based on set_type
            if edge['type'] == 'inter' or edge.get('set_type') == 'train':
                set_type_edges['train'].append(pair)
            elif edge.get('set_type') == 'val':
                set_type_edges['val'].append(pair)
            else:
                set_type_edges['test'].append(pair)
        return edges, set_type_edges

    def generate_edges(self, negative_sample_ratio=1.0, threshold=1):
        train_edges = self.set_type_edges['train']
        val_edges = self.set_type_edges['val']
        test_edges = self.set_type_edges['test']
        return train_edges, val_edges, test_edges

