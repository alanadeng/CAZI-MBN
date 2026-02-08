from sklearn.model_selection import train_test_split
import pandas as pd

class DataSplitter:
    def __init__(self, data, split_ratio=(8, 1.0, 1.0), mode='transductive', seed=0):
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
        # Split data based on specified ratio, ensuring all nodes appear in the training set
        train_size = self.split_ratio[0] / sum(self.split_ratio)
        #val_test_ratio = self.split_ratio[1] / (self.split_ratio[1] + self.split_ratio[2])

        train_data, temp_data = train_test_split(self.data, train_size=train_size, random_state=self.seed)
        #val_data, test_data = train_test_split(temp_data, train_size=val_test_ratio, random_state=self.seed)

        # print(train_data, val_data, test_data)

        #return train_data, val_data, test_data
        return train_data, temp_data, temp_data


    def _inductive_split(self):
        unique_drugs = self.data['drug'].unique()
        single_entry_drugs = self.data['drug'].value_counts() == 1  # Drugs with a single entry
        multi_entry_drugs = unique_drugs[~single_entry_drugs]

        # Split multi-entry drugs for training
        train_drugs, temp_drugs = train_test_split(multi_entry_drugs, train_size=self.split_ratio[0] / sum(self.split_ratio), random_state=self.seed)
        val_drugs, test_drugs = train_test_split(temp_drugs, train_size=0.5, random_state=self.seed)

        # Always add single-entry drugs to training
        train_drugs = pd.concat([train_drugs, unique_drugs[single_entry_drugs.index]])

        train_data = self.data[self.data['drug'].isin(train_drugs)]
        val_data = self.data[self.data['drug'].isin(val_drugs)]
        test_data = self.data[self.data['drug'].isin(test_drugs)]

        return train_data, val_data, test_data