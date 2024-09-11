import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
data_cleaned = pd.read_csv('history.csv', encoding='ISO-8859-1')

# Convert columns to numeric if needed
numeric_columns = ['Visit Count', 'URL Length', 'Typed Count']
for column in numeric_columns:
    if column in data_cleaned.columns:
        data_cleaned[column] = pd.to_numeric(data_cleaned[column], errors='coerce')

# Ensure 'Visit Time' is a datetime object
if 'Visit Time' in data_cleaned.columns:
    data_cleaned['Visit Time'] = pd.to_datetime(data_cleaned['Visit Time'], errors='coerce')
    data_cleaned['Visit Date'] = data_cleaned['Visit Time'].dt.date
    data_cleaned['Visit Hour'] = data_cleaned['Visit Time'].dt.hour

# 1. Bar Charts/Histograms: Most Visited Websites or Categories
if 'URL' in data_cleaned.columns:
    top_urls = data_cleaned['URL'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_urls.values, y=top_urls.index, palette='viridis')
    plt.title("Top 10 Most Frequently Visited URLs")
    plt.xlabel("Number of Visits")
    plt.ylabel("URL")
    plt.show()

# 2. Heatmaps: Visit Patterns Over Time (Peak Browsing Hours)
if 'Visit Hour' in data_cleaned.columns:
    visits_by_hour = data_cleaned.groupby(['Visit Date', 'Visit Hour']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(visits_by_hour, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Number of Visits'})
    plt.title("Heatmap of Visits by Hour")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Date")
    plt.show()

# 3. Line Charts: Changes in Visit Duration or Frequency Over Time
if 'Visit Duration' in data_cleaned.columns:
    # Convert 'Visit Duration' to numeric, handling time format if necessary
    data_cleaned['Visit Duration'] = pd.to_timedelta(data_cleaned['Visit Duration'], errors='coerce')
    daily_duration = data_cleaned.groupby('Visit Date')['Visit Duration'].sum()
    plt.figure(figsize=(12, 6))
    daily_duration.plot()
    plt.title("Total Visit Duration Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Visit Duration")
    plt.show()

if 'Visit Count' in data_cleaned.columns:
    daily_visits = data_cleaned.groupby('Visit Date')['Visit Count'].sum()
    plt.figure(figsize=(12, 6))
    daily_visits.plot()
    plt.title("Total Visits Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Number of Visits")
    plt.show()
