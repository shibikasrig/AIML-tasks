
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.describe(include="all"))

print(df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin or make new binary feature
df['CabinKnown'] = df['Cabin'].notnull().astype(int)
df.drop(columns=['Cabin'], inplace=True)

# Drop Ticket or process separately (complex string data)
df.drop(columns=['Ticket'], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x=df['Age'])
plt.title("Age Outliers")

plt.subplot(1,2,2)
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()

# Define a function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply to Age and Fare
df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')

print("After outlier removal:", df.shape)