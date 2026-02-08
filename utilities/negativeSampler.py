import pandas as pd
from itertools import product

class NegativeSampler:
    def __init__(self, data, ratio=0.25):
        """
        Initialize the NegativeSampler with data and a negative sampling ratio.

        Parameters:
            data (pd.DataFrame): The dataset containing columns 'drug', 'gene', and class labels '0', '1', ...
            ratio (float): The ratio of negative samples to positive samples.
        """
        self.data = data
        self.ratio = ratio
        self.drug_ids = data['drug'].unique()
        self.gene_ids = data['gene'].unique()

    def generate_negative_samples(self):
        """
        Generate negative samples for unseen drug-gene combinations.

        Returns:
            pd.DataFrame: A DataFrame containing negative samples with all class labels set to 0.
        """
        # Calculate the number of negative samples to generate
        num_positives = len(self.data)
        num_negatives = int(self.ratio * num_positives)

        # Create all possible combinations of drug and gene IDs
        all_combinations = pd.DataFrame(list(product(self.drug_ids, self.gene_ids)), columns=['drug', 'gene'])

        # Find existing combinations in the data
        existing_combinations = self.data[['drug', 'gene']].drop_duplicates()

        # Use merge to find combinations not in the original data
        negative_samples = pd.merge(all_combinations, existing_combinations, on=['drug', 'gene'], how='left', indicator=True)
        negative_samples = negative_samples[negative_samples['_merge'] == 'left_only'].drop('_merge', axis=1)

        # Randomly select the needed number of negatives
        negative_samples = negative_samples.sample(n=num_negatives, random_state=42)

        # Add label columns and set them to 0
        for col in self.data.columns[2:]:
            negative_samples[col] = 0

        return negative_samples

    def combine_samples(self):
        """
        Generate negative samples and combine them with the original dataset.

        Returns:
            pd.DataFrame: The original dataset combined with the generated negative samples.
        """
        negative_samples = self.generate_negative_samples()
        combined_data = pd.concat([self.data, negative_samples], ignore_index=True)
        return combined_data