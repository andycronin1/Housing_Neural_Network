import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime

from dask.array import average
from dotenv import load_dotenv
from pandas.io.sas.sas_constants import column_name_text_subheader_length
from sqlalchemy.cyextension.resultproxy import rowproxy_reconstructor
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

        # dummy_data = self.data[['loan_status', 'emp_length']]
        # dummy_data['Numerated'] = [True if x=='Charged Off' else False for x in dummy_data['loan_status']]
        #
        # percentage_data = dummy_data.groupby('emp_length')['Numerated'].mean().reset_index()
        # percentage_data.set_index('emp_length', inplace=True)
        # sns.barplot(data=percentage_data, x='emp_length', y='Numerated', order=emp_length_order, palette='coolwarm')
        # # plt.show()

        # Charge off rates are extremely similar across all employment lengths. So we can drop the emp_length column.
        self.data.drop(columns=['emp_length'], inplace=True)


        # The title column is simply a string subcategory/description of the purpose column. so we can drop the title column.
        self.data.drop(columns=['title'], inplace=True)

        # mort_acc review and missing data fill
        print(f' Mort_acc description: {self.info_data.loc['mort_acc']['Description']}')

        mort_acc_val_counts = self.data['mort_acc'].value_counts()

        correlation_matrix = self.data.corr(numeric_only=True)

        # Let's fill in the missing mort_acc values based on their total_acc value as the total_acc value correlates the best with mort_acc.

        average_mort_acc = self.data.groupby('total_acc')['mort_acc'].mean()

        # Create a function to fill missing mort_acc values based on total_acc

        def fill_mort_acc(total_acc, mort_acc):

            if pd.isna(mort_acc):
                return average_mort_acc.loc[total_acc]
            else:
                return mort_acc

        # Apply the function to fill missing mort_acc values
        self.data['mort_acc'] = self.data.apply(lambda row: fill_mort_acc(row['total_acc'], row['mort_acc']), axis=1)

        # revol_util and pub_rec_bankruptcies account for less than 0.5% of the total data so we can remove the rows that are missing those values.
        self.data.dropna(subset=['revol_util', 'pub_rec_bankruptcies'], inplace=True)

        print(self.data.isnull().sum())

        # Clraning completed

    def run(self):
        pass

if __name__ == "__main__":
    pipeline = DataCleaningPipeline()
    pipeline.analyse_nulls()
    print("Data cleaning pipeline completed.")
    # You can add more methods to the class to perform specific cleaning tasks.
    # For example, you could add methods to handle missing values, outliers, etc.

