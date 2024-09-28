import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

# API URL for fetching data from the UCI Machine Learning Repository
API_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart.csv"  # UCI Heart Disease dataset

# Column names as provided
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Fetch data from the API
try:
    response = requests.get(API_URL)
    if response.status_code == 200:
        # Decode the batch data directly into a DataFrame
        data = response.content.decode('utf-8')
        df = pd.read_csv(pd.compat.StringIO(data), names=column_names, header=0)  # Create DataFrame with defined columns
        print(f"Data fetched from API: {len(df)} records.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        exit()
except Exception as e:
    print(f"Error fetching data from API: {e}")
    exit()

# Basic Data Information
print("\n--- Basic Information ---")
print(df.info())
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Summary Statistics ---")
print(df.describe())

# Check for Missing Values
print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())

# Visualize Distributions of Numerical Features
num_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df[num_columns].hist(bins=15, figsize=(15, 10), layout=(3, 2))
plt.tight_layout()
plt.show()

# Visualize the distribution of Categorical Features
cat_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_columns):
    plt.subplot(3, 3, i + 1)
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Feature Engineering
num_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
scaler = StandardScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])

cat_columns = ["cp", "restecg", "slope", "ca", "thal"]
encoder = LabelEncoder()
for col in cat_columns:
    df[col] = encoder.fit_transform(df[col])

# Save the cleaned data for future use in the decision tree classifier
df.to_csv("cleaned_heart_disease_data.csv", index=False)
print("Preprocessed data saved.")
