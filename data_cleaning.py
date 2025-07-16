import os
import pandas as pd
from dotenv import load_dotenv
from Database_Connector import DatabaseConnector

load_dotenv()

BASE_DIR = os.getenv('BASE')

# TODO: Export the csv data to a postgresql database and import data from there

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

        # Cleaning completed

    def string_value_fixing(self):

        # Listing all non-numeric values
        print(self.data.select_dtypes(['object']).columns)

        # Convert the term feature into a integer numeric data type.
        self.data['term'] = self.data['term'].apply(lambda term: int(term[:3]))

        ### We already know grade is part of sub_grad, so we can just drop the grade feature
        self.data.drop(columns=['grade'], axis=1, inplace=True)

        # Convert the subgrade into dummy variables
        subgrade_dummies = pd.get_dummies(self.data['sub_grade'], drop_first=True)

        self.data = pd.concat([self.data, subgrade_dummies], axis = 1)
        self.data.drop(columns=['sub_grade'], axis=1, inplace=True)

        # Converting ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables
        other_dummies = pd.get_dummies(self.data[['verification_status', 'application_type','initial_list_status','purpose']], drop_first=True)
        self.data = pd.concat([self.data, other_dummies], axis=1)
        self.data.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1, inplace=True)

        # converting home_ownership to dummy variables, but replacing NONE and ANY with OTHER.
        self.data['home_ownership'].apply(lambda x: x == 'OTHER' if x == 'NONE' or 'ANY' else x)
        print(self.data['home_ownership'].value_counts)
        self.data.drop(columns='home_ownership', axis=1, inplace=True)

        # Feature engineering a zip code column from the address column.
        self.data['zip_code'] = self.data['address'].apply(lambda x : x[-5:])
        # Make zip code dummy variables
        zip_dummies = pd.get_dummies(self.data['zip_code'], drop_first=True)
        self.data = pd.concat([self.data, zip_dummies], axis=1)
        self.data.drop(columns=['zip_code', 'address'], axis=1, inplace=True)

        # Dropping issue_d column
        self.data.drop('issue_d', axis=1, inplace=True)

        self.data['earliest_cr_line'] = pd.to_numeric(self.data['earliest_cr_line'].apply(lambda x: x[-4:]))


        # feature engineering completed.

    def final_fixes(self):
        # create loan repaid column. 1 for fully paid. 0 for charged off.
        self.data['loan_repaid'] = self.data['loan_status'].apply(lambda x: 1 if x == 'Fully Paid' else 0)
        self.data.drop(columns='loan_status', axis=1, inplace=True)

    def fetch_data(self):
        db_connector = DatabaseConnector(db_name='ML_data')
        data = db_connector.fetch_all_data()
        a=1


    def run(self):
        self.analyse_nulls()
        self.string_value_fixing()
        self.final_fixes()
        print("Data cleaning pipeline completed.")

if __name__ == "__main__":
    pipeline = DataCleaningPipeline()
    # pipeline.analyse_nulls()
    # pipeline.string_value_fixing()
    # pipeline.final_fixes()
    # print("Data cleaning pipeline completed.")
    pipeline.fetch_data()
    # You can add more methods to the class to perform specific cleaning tasks.
    # For example, you could add methods to handle missing values, outliers, etc.

