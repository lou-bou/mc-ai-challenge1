import pandas as pd
import matplotlib
matplotlib.use("TkAgg") # set backend for interfaces
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# load train.csv
data = pd.read_csv('./train.csv')

# display first 10 rows
first_ten_rows = data.head(10)

# dataset shape and column information
data_shape = data.shape

#column_info = data.info()

# descriptive stats
desc_stats = data.describe()

# value counts for categorical columns
names = np.array(data['Name'].unique())
sexes = np.array(data['Sex'].unique())
embarked_cats = np.array(data['Embarked'].unique())

# create column
def set_age_range(age):
    if age <= 30:
        return "Young"
    elif age >= 60:
        return "Old"
    elif age > 30 and age < 60:
        return "Middle-age"
    else: # for NaN (before cleaning)
        return "NA"

data['Age_range'] = data['Age'].apply(set_age_range)

# bar chart of number of people per sex
sex_cats = np.array(['Female', 'Male'])
sex_values = np.array(
                data.groupby('Sex')['PassengerId']
                .sum()
                .reset_index(name ='Sum')
                ['Sum'])

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

axes[0].bar(sex_cats, sex_values, color='green')
axes[0].set_title("Number of people per sex")

# histogram of ages
axes[1].hist(np.array(data['Age']), bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title("Number of people per age")

plt.tight_layout()
#plt.show()

# Rows with missing values
data_missing = data.isnull()
total_missing = data.isnull().sum()

# filling missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Cabin'] = data['Cabin'].fillna(0) # ill probably drop this later anyways so it doesnt matter

#print(data.isnull().sum()) returns 0 null values

# encoding sex
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# encoding embarked
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# encoding age_range
age_mapping = {'Young': 0, 'Middle-age': 1, 'Old': 2}
data['Age_range'] = data['Age_range'].map(age_mapping)

# drop ticket, name and cabin because they're identifiers and useless
data = data.drop('Ticket', axis=1)
data = data.drop('Cabin', axis=1)
data = data.drop('Name', axis=1)