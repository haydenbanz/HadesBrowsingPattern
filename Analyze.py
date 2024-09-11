import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data with a specified encoding
data_cleaned = pd.read_csv('history.csv', encoding='ISO-8859-1')

# Inspect column names
print("Column names in the dataset:")
print(data_cleaned.columns)

# Convert columns to numeric if needed
# Ensure the column names match those in the dataset
numeric_columns = ['Visit Count', 'URL Length', 'Typed Count']
for column in numeric_columns:
    if column in data_cleaned.columns:
        data_cleaned[column] = pd.to_numeric(data_cleaned[column], errors='coerce')

# Check for numeric columns after conversion
numeric_columns_found = data_cleaned.select_dtypes(include=[np.number]).columns
print("\nNumeric columns found:")
print(numeric_columns_found)

# 1. Analyzing Visit Patterns Over Time
if 'Visit Time' in data_cleaned.columns:
    data_cleaned['Visit Time'] = pd.to_datetime(data_cleaned['Visit Time'], errors='coerce')
    data_cleaned['Visit Date'] = data_cleaned['Visit Time'].dt.date
    data_cleaned['Visit Hour'] = data_cleaned['Visit Time'].dt.hour

    # Plotting the number of visits over time
    visit_counts = data_cleaned.groupby('Visit Date').size()
    plt.figure(figsize=(12, 6))
    visit_counts.plot()
    plt.title("Number of Visits Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Visits")
    plt.show()

    # Plotting the distribution of visits by hour
    plt.figure(figsize=(12, 6))
    sns.histplot(data_cleaned['Visit Hour'].dropna(), bins=24, kde=False)
    plt.title("Distribution of Visits by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Visits")
    plt.show()

# 2. Analyzing the Frequency of URL Visits
if 'URL' in data_cleaned.columns:
    top_urls = data_cleaned['URL'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_urls.values, y=top_urls.index, palette='viridis')
    plt.title("Top 10 Most Frequently Visited URLs")
    plt.xlabel("Number of Visits")
    plt.ylabel("URL")
    plt.show()

# 3. Analyzing Sentiment Distribution (if available)
if 'Sentiment' in data_cleaned.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Sentiment', data=data_cleaned, palette='coolwarm')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.show()

# 4. Analyzing Correlation Between Features
if len(numeric_columns_found) > 0:
    correlation_matrix = data_cleaned[numeric_columns_found].corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()
else:
    print("No numeric columns found for correlation analysis.")
