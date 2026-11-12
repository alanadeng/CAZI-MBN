class GraphHandler:
    def __init__(self, train_data, val_data, test_data=None, num_layer=6, interaction_columns=None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.nodes = []
        self.edges = []
        self.id_counter = 0
        self.entity_id_map = {}
        self.entity_layers_map = {}
        self.train_entity_layers_map = {}
        self.num_layer = num_layer
        self.interaction_columns = interaction_columns

    def get_unique_id(self, entity, layer, entity_type, set_type=None):
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
        if set_type == 'train':
            if entity_key not in self.train_entity_layers_map:
                self.train_entity_layers_map[entity_key] = []
            if layer not in self.train_entity_layers_map[entity_key]:
                self.train_entity_layers_map[entity_key].append(layer)
        return self.entity_id_map[key]

    def generate_edges(self, data, set_type):
        if data is None:
            return
        if self.interaction_columns is None:
            interaction_columns = [str(label) for label in range(self.num_layer)]
        else:
            interaction_columns = self.interaction_columns
        if len(interaction_columns) != self.num_layer:
            raise ValueError(
                f"Expected {self.num_layer} interaction columns, "
                f"got {len(interaction_columns)}."
            )
        for _, row in data.iterrows():
            for layer, column in enumerate(interaction_columns):
                if row[column] == 1:
                    gene_id = self.get_unique_id(row['gene'], layer, "gene", set_type)
                    drug_id = self.get_unique_id(row['drug'], layer, "drug", set_type)
                    self.edges.append({
                        "source": drug_id,
                        "target": gene_id,
                        "layer": layer,
                        "type": "intra",
                        "set_type": set_type
                    })

    def add_inter_layer_edges(self):
        for entity_key, layers in self.train_entity_layers_map.items():
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
        # Register edges from every split; only training edges enter the encoder.
        self.generate_edges(self.train_data, 'train')
        self.generate_edges(self.val_data, 'val')
        self.generate_edges(self.test_data, 'test')
        self.add_inter_layer_edges()

        # The graph is now represented as a combination of nodes and edges
        return {"nodes": self.nodes, "edges": self.edges}