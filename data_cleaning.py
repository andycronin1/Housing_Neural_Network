import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime

from dotenv import load_dotenv
from pandas.io.sas.sas_constants import column_name_text_subheader_length
from tensorflow.python.ops.ragged.ragged_embedding_ops import embedding_lookup_sparse

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

    def feat_info(self, *args):
        for arg in args:
            print(f' Description of {arg} is: {self.info_data.loc[arg]['Description']}')

    def analyse_nulls(self):
        # Analysing nulls as percentage of the dataframe
        print(self.data.isnull().sum())
        print(100* self.data.isnull().sum()/len(self.data))

        # Understanding the data in high percentage nulls columns.
        self.feat_info('emp_title', 'emp_length')
        self.feat_info()

        print(f'Number of unique Jobs titles in the dataset = {self.data['emp_title'].nunique()}')
        # Realistically there are too many unique job titles to try to convert this to a dummy variable feature. So remove the emp_title column.
        self.data.drop(columns=['emp_title'], inplace=True)

        emp_length_order = ['< 1 year',
                            '1 year',
                            '2 years',
                            '3 years',
                            '4 years',
                            '5 years',
                            '6 years',
                            '7 years',
                            '8 years',
                            '9 years',
                            '10+ years']

        # sns.countplot(data=self.data, x='emp_length', order=emp_length_order, palette='coolwarm', hue='loan_status')
        # plt.show()

        # **CHALLENGE TASK: \
        # This still doesn't really inform us if there is a strong relationship between employment length and being charged off,
        # what we want is the percentage of charge offs per category.
        # Essentially informing us what percent of people per employment length category didn't pay back their loan.
        # There are a multitude of ways to create this Series.
        # Once you've created it, see if visualize it with a [bar plot](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html).
        # This may be tricky, refer to solutions if you get stuck on creating this Series.**s

        dummy_data = self.data[['loan_status', 'emp_length']]
        dummy_data['Numerated'] = [True if x=='Charged Off' else False for x in dummy_data['loan_status']]

        percentage_data = dummy_data.groupby('emp_length')['Numerated'].mean().reset_index()
        percentage_data.set_index('emp_length', inplace=True)
        sns.barplot(data=percentage_data, x='emp_length', y='Numerated', order=emp_length_order, palette='coolwarm')
        plt.show()



        a=1



    def run(self):
        pass

if __name__ == "__main__":
    pipeline = DataCleaningPipeline()
    pipeline.analyse_nulls()
    print("Data cleaning pipeline completed.")
    # You can add more methods to the class to perform specific cleaning tasks.
    # For example, you could add methods to handle missing values, outliers, etc.

