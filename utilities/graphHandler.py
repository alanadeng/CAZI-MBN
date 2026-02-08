class GraphHandler:
    def __init__(self, train_data, val_data, test_data=None, num_layer = 6):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.nodes = []
        self.edges = []
        self.id_counter = 0
        self.entity_id_map = {}
        self.entity_layers_map = {}
        self.num_layer = num_layer

    def get_unique_id(self, entity, layer, entity_type):
        key = f"{entity_type}_{entity}_layer_{layer}"
        if key not in self.entity_id_map:
            self.entity_id_map[key] = self.id_counter
            self.nodes.append({"id": self.id_counter, "label": entity, "type": entity_type, "layer": layer})
            self.id_counter += 1
        entity_key = f"{entity_type}_{entity}"
        if entity_key not in self.entity_layers_map:
            self.entity_layers_map[entity_key] = []
        if layer not in self.entity_layers_map[entity_key]:
            self.entity_layers_map[entity_key].append(layer)
        return self.entity_id_map[key]

    def generate_edges(self, data, set_type):
        for _, row in data.iterrows():
            for label in range(self.num_layer):
                if row[str(label)] == 1:
                    gene_id = self.get_unique_id(row['gene'], label, "gene")
                    drug_id = self.get_unique_id(row['drug'], label, "drug")
                    self.edges.append({
                        "source": drug_id,
                        "target": gene_id,
                        "layer": label,
                        "type": "intra",
                        "set_type": set_type
                    })

    def add_inter_layer_edges(self):
        for entity_key, layers in self.entity_layers_map.items():
            layers = sorted(set(layers))
            for i in range(len(layers) - 1):
                for j in range(i + 1, len(layers)):
                    entity_type, entity = entity_key.split("_", 1)
                    id_i = self.entity_id_map[f"{entity_type}_{entity}_layer_{layers[i]}"]
                    id_j = self.entity_id_map[f"{entity_type}_{entity}_layer_{layers[j]}"]
                    self.edges.append({
                        "source": id_i,
                        "target": id_j,
                        "type": "inter",
                        "from_layer": layers[i],
                        "to_layer": layers[j]
                    })

    def generate_graph(self):
        # Process both training and validation datasets
        self.generate_edges(self.train_data, 'train')
        self.generate_edges(self.val_data, 'val')
        self.generate_edges(self.test_data, 'test')
        # Add inter-layer edges to connect the same entities across different contexts
        self.add_inter_layer_edges()

        # The graph is now represented as a combination of nodes and edges
        return {"nodes": self.nodes, "edges": self.edges}