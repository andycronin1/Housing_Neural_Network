import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.getenv('BASE')

class DataCleaningPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = BASE_DIR
        else:
            self.base_dir = base_dir

        self.data_file_path = os.path.join(self.base_dir, 'DATA', 'lending_club_loan_two.csv')
        self.data_info_path = os.path.join(self.base_dir, 'DATA', 'lending_club_info.csv')
        self.data = pd.read_csv(self.data_file_path)
        self.info_data = pd.read_csv(self.data_info_path, index_col='LoanStatNew')

    def feat_info(self, col_name):
        print(self.info_data.loc[col_name]['Description'])

    def analyse_nulls(self):
        # Analysing nulls as percentage of the dataframe
        print(self.data.isnull().sum())
        print(100* self.data.isnull().sum()/len(self.data))

        filled_data = self.data.dropna()
        print(f'Number of NaN values dropped = {self.data.shape[0] - filled_data.shape[0]}')

    def run(self):
        pass

if __name__ == "__main__":
    pipeline = DataCleaningPipeline()
    pipeline.analyse_nulls()
    print("Data cleaning pipeline completed.")
    # You can add more methods to the class to perform specific cleaning tasks.
    # For example, you could add methods to handle missing values, outliers, etc.

