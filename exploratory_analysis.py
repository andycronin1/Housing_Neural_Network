import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.getenv('BASE')

print(BASE_DIR)

data_file_path = os.path.join(BASE_DIR, 'DATA', 'lending_club_loan_two.csv')
data_info_path = os.path.join(BASE_DIR, 'DATA', 'lending_club_info.csv')
data = pd.read_csv(data_file_path)
info_data = pd.read_csv(data_info_path, index_col='LoanStatNew')

# Data Exploration
print(data['loan_status'].value_counts())

# Plotting the distribution of loan statuses
# plt.bar(data['loan_status'].value_counts().index,  data['loan_status'].value_counts().values)
# plt.xlabel('Loan Status')
# plt.ylabel('Count')
# plt.show()

# Displaying histogram of loan amounts
# plt.hist(data['loan_amnt'], bins=100, edgecolor='black')
# plt.xlabel('Loan Amount')
# plt.ylabel('Frequency')
# plt.show()

# Calculate the correlation between all continuous numeric variables using .corr() method
correlation_matrix = data.corr(numeric_only=True)
print(correlation_matrix)

# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.show()

print(f'Installement Description: {info_data.loc['installment']['Description']}')
print(f'Loan Amount Description: {info_data.loc['loan_amnt']['Description']}')

# sns.scatterplot(data=data, x='installment', y='loan_amnt', hue='loan_status', alpha=0.5)
# plt.show()

data_summary_stats = data.groupby('loan_status')['loan_amnt'].agg(['mean', 'std', 'min', 'max', 'count'])
print(data_summary_stats)

# Explore the Grade and SubGrade columns that LendingClub attributes to the loans. What are the unique possible grades and subgrades?
unique_grades = data['grade'].unique()
unique_subgrades = data['sub_grade'].unique()
print(f'Unique Grades: {unique_grades}')
print(f'Unique Subgrades: {unique_subgrades}')

# count plot of subgrade with frequency. Shows the distribution of subgrades across different loan statuses
plt.figure(figsize=(12,4))
sorted_subgrades = sorted(data['sub_grade'].unique())
sns.countplot(data, x='sub_grade', order=sorted_subgrades, palette='coolwarm', hue='loan_status')
plt.show()

grouped_data = data.groupby(['sub_grade', 'emp_title'])['installment'].agg(['mean', 'std', 'min', 'max', 'count'])
print(grouped_data)

a=1