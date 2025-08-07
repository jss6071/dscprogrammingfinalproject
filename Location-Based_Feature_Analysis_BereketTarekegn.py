# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 20:28:42 2025

@author: Bereke Tarekegn
"""

# Step 1: Import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Show plots in Spyder's plot pane
%matplotlib inline

# Step 2: upload file 
os.chdir(r"C:\Users\bezaa\OneDrive\Desktop\USD\Data Science Programming (ADS-500B-01\Group Assignment\My Project\Dataset 2 (House Sales)")

# Step 2.1: Load the dataset
df = pd.read_csv("house_sales.csv")

# Step 2.2: Show the first few rows
print(df.head())

# Step 3: location-based features and price
location_df = df[['zipcode', 'lat', 'long', 'price']]

# See data types and if anything is missing
print(location_df.info())

# Basic statistics
print(location_df.describe())

# Count missing values
print(location_df.isnull().sum())

#4.1. Listings per Zipcode
plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='zipcode', order=df['zipcode'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Number of Listings per Zipcode")
plt.xlabel("Zipcode")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#4.2. Map of House Locations Colored by Price
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='long', y='lat', hue='price', palette='coolwarm', alpha=0.6)
plt.title("House Locations Colored by Price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

#4.3. Correlation Between Location and Price
# Correlation matrix for lat, long, and price
print(location_df.corr())

#Step 5: Regression with Location Features
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Use only lat and long to predict price
X = df[['lat', 'long']]
y = df['price']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Check R² score (how well location explains price)
r2_score = model.score(X_test, y_test)
print(f"R² Score using lat/long to predict price: {r2_score:.4f}")

