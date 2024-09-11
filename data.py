import pandas as pd
import numpy as np

# Step 1: Load the data
# Use a specific encoding to handle any special characters
try:
    data = pd.read_csv('history.csv', encoding='ISO-8859-1')
except UnicodeDecodeError as e:
    print("Error loading CSV file:", e)
    print("Trying with a different encoding...")
    data = pd.read_csv('history.csv', encoding='latin1')

# Step 2: Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Step 3: Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(data.head())

# Step 4: Handling missing values
# Optionally, drop rows with any missing values
data_cleaned = data.dropna()

# Alternatively, fill missing values with a placeholder
# data_cleaned = data.fillna("Unknown")

print("\nNumber of rows before cleaning:", len(data))
print("Number of rows after cleaning:", len(data_cleaned))

# Step 5: Basic preprocessing
# Example: Converting text to lowercase (assuming a column 'text')
if 'text' in data_cleaned.columns:
    data_cleaned['text'] = data_cleaned['text'].str.lower()

# Example: Removing special characters from text
import re
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

if 'text' in data_cleaned.columns:
    data_cleaned['text'] = data_cleaned['text'].apply(clean_text)

# Step 6: Save the cleaned data for further analysis
data_cleaned.to_csv('history_cleaned.csv', index=False)
print("\nCleaned data saved to 'history_cleaned.csv'.")

# Step 7: Basic analysis (example)
# Assuming we want to see the distribution of a specific column 'sentiment'
if 'sentiment' in data_cleaned.columns:
    print("\nSentiment Distribution:")
    print(data_cleaned['sentiment'].value_counts())

# Further steps might involve more complex processing, such as sentiment analysis, emotion detection, etc.
