from sklearn.model_selection import train_test_split
import pandas as pd

class DataSplitter:
    def __init__(self, data, split_ratio=(70, 15, 15), mode='transductive', seed=0):
        self.split_ratio = split_ratio
        self.mode = mode
        self.data = data
        self.seed = seed

    def split(self):
        if self.mode == 'transductive':
            return self._transductive_split()
        elif self.mode == 'inductive':
            return self._inductive_split()
        else:
            raise ValueError("Mode must be either 'transductive' or 'inductive'.")

    def _transductive_split(self):
        train_ratio, val_ratio, test_ratio = self._normalized_split_ratio()
        train_data, temp_data = train_test_split(
            self.data,
            train_size=train_ratio,
            random_state=self.seed,
        )

        train_drugs = set(train_data['drug'])
        train_genes = set(train_data['gene'])
        missing_node_mask = (
            ~temp_data['drug'].isin(train_drugs)
            | ~temp_data['gene'].isin(train_genes)
        )
        train_data = pd.concat([train_data, temp_data.loc[missing_node_mask]])
        temp_data = temp_data.loc[~missing_node_mask]

        if len(temp_data) < 2:
            raise ValueError(
                "A transductive split needs at least two held-out interaction rows "
                "whose entities are already present in training."
            )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_size,
            random_state=self.seed,
        )
        return train_data, val_data, test_data

    def _inductive_split(self):
        unique_drugs = self.data['drug'].unique()
        unique_genes = self.data['gene'].unique()
        train_ratio, val_ratio, test_ratio = self._normalized_split_ratio()

        def split_entities(entities):
            if len(entities) < 3:
                return set(entities), set(), set()
            train_entities, temp_entities = train_test_split(
                entities,
                train_size=train_ratio,
                random_state=self.seed,
            )
            val_entities, test_entities = train_test_split(
                temp_entities,
                train_size=val_ratio / (val_ratio + test_ratio),
                random_state=self.seed,
            )
            return (
                set(train_entities),
                set(val_entities),
                set(test_entities),
            )

        shared = set(unique_drugs).intersection(unique_genes)
        drug_only = [entity for entity in unique_drugs if entity not in shared]
        gene_only = [entity for entity in unique_genes if entity not in shared]
        shared_entities = [entity for entity in unique_drugs if entity in shared]

        _, val_drugs, test_drugs = split_entities(drug_only)
        _, val_genes, test_genes = split_entities(gene_only)
        _, val_shared, test_shared = split_entities(shared_entities)
        val_entities = val_drugs | val_genes | val_shared
        test_entities = test_drugs | test_genes | test_shared

        has_test = (
            self.data['drug'].isin(test_entities)
            | self.data['gene'].isin(test_entities)
        )
        has_val = (
            self.data['drug'].isin(val_entities)
            | self.data['gene'].isin(val_entities)
        )
        test_mask = has_test & ~has_val
        val_mask = has_val & ~has_test
        train_mask = ~has_test & ~has_val

        train_data = self.data.loc[train_mask]
        val_data = self.data.loc[val_mask]
        test_data = self.data.loc[test_mask]

        return train_data, val_data, test_data

    def _normalized_split_ratio(self):
        if len(self.split_ratio) != 3:
            raise ValueError("split_ratio must contain train, validation, and test proportions.")
        if any(ratio <= 0 for ratio in self.split_ratio):
            raise ValueError("All split_ratio values must be greater than zero.")

        total = sum(self.split_ratio)
        return tuple(ratio / total for ratio in self.split_ratio)