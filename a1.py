import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Import the dataset and explore basic info
df = pd.read_csv('Titanic-Dataset.csv')

# # Display basic info
print(df.info())

# # Display null values
print(df.isnull().sum())

# # Display data types
print(df.dtypes)

# # 2. Handle missing values
# # For Age - fill with median
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)

# # For Cabin - too many missing values, we'll drop this column
df.drop('Cabin', axis=1, inplace=True)

# # For Embarked - fill with mode
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)

# # For Fare - fill with median (though no missing values in this dataset)
fare_median = df['Fare'].median()
df['Fare'].fillna(fare_median, inplace=True)

# # Verify no more null values
print(df.isnull().sum())

# # 3. Convert categorical features to numerical
# # Label encoding for Sex and Embarked
label_encoders = {}
categorical_cols = ['Sex', 'Embarked']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# # One-hot encoding for Pclass (since it's ordinal but not strictly numeric)
df = pd.get_dummies(df, columns=['Pclass'], prefix='Pclass')

# # Drop columns that won't be useful for modeling
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# # 4. Normalize/standardize numerical features
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# # 5. Visualize and handle outliers
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, len(numerical_cols), i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# # Function to remove outliers using IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# # Remove outliers from numerical columns
for col in numerical_cols:
    df = remove_outliers(df, col)

# # Display final dataset info
print(df.info())
print(df.head())