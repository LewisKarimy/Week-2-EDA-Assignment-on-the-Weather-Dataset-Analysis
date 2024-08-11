
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from scipy import stats


# Step 1: Data Overview and Cleaning

# Check shape and data types
df = pd.read_csv('1. Weather Data.csv')
print(df.shape) #prints the total number of rows and columns in the dataset
print(df.dtypes) 

#Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values (replace with appropriate strategy)
print(df.fillna(method='ffill', inplace=True))  # No missing value

#Check for duplicates
print(df.duplicated().sum()) #no duplicates


# Step 2. Statistical Summary

print("\nStatistical Summary:")
print(df.describe())


#Identify potential outliers (using IQR) -They are extreme values in a dataset that significantly differ from the majority of the data points. 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Determine outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#Identify outliers

z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).any(axis=1)

print("\nNumber of outliers detected:")
print(outliers.sum())

outlier_rows = df[outliers]
print("\nOutliers in the dataset:")
print(outlier_rows.describe())


# Step 3. Data Visualization

#Distribution of key weather parameters
sns.histplot(df['Temp_C'])
plt.title('Temparature Distribution')
plt.show()

sns.histplot(df['Rel Hum_%'])
plt.title("Humidity Distribution")
plt.show()

sns.histplot(df['Wind Speed_km/h'])
plt.title("Wind Speed Distribution")
plt.show()

sns.histplot(df['Press_kPa'])
plt.title('Pressure Distribution')
plt.show()


#Time series plots

#Ensure the 'Date' column is in datetime format
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Set 'Date' as the index
df.set_index('Date/Time', inplace=True)

# Plot time series graphs
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
df['Temp_C'].plot()
plt.title('Temperature Over Time')

plt.subplot(1, 3, 2)
df['Rel Hum_%'].plot()
plt.title('Humidity Over Time')

plt.subplot(1, 3, 3)
df['Wind Speed_km/h'].plot()
plt.title('Wind Speed Over Time')

plt.tight_layout()
plt.show()


# Convert the datetime column to datetime type
# Replace 'Date' with the actual name of your date column
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract numerical features from the datetime
df['Year'] = df['Date/Time'].dt.year
df['Month'] = df['Date/Time'].dt.month
df['Day'] = df['Date/Time'].dt.day
df['Hour'] = df['Date/Time'].dt.hour

# Select only the numerical columns for correlation analysis
numerical_columns = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 
                     'Visibility_km', 'Press_kPa', 'Year', 'Month', 'Day', 'Hour']

# Ensure all selected columns are numeric
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with NaN values
data_numeric = df[numerical_columns].dropna()

# Calculate the correlation matrix
corr_matrix = data_numeric.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)

# Set the title
plt.title('Correlation Heatmap of Weather Parameters')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Print the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# Optional: Create pairplot for more detailed view
sns.pairplot(data_numeric)
plt.tight_layout()
plt.show()


# Step 4. Weather Patterns and Trends

#Seasonal analysis (assuming 'date' column has a datetime format)
df['month'] = df['date'].dt.month
df.groupby('month')[['temperature', 'humidity']].mean().plot(kind='bar')
plt.show()


# Anomaly detection (example using z-score)

# Convert the datetime column to datetime type
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Set the date as the index
df.set_index('Date/Time', inplace=True)

# List of numerical columns to analyze
numerical_columns = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 
                     'Visibility_km', 'Press_kPa']

# Function to detect outliers using Z-score
def detect_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores > threshold]

# Function to plot time series with outliers highlighted
def plot_time_series_with_outliers(df, column):
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[column], label='Data')
    outliers = detect_outliers(df, column)
    plt.scatter(outliers.index, outliers[column], color='red', label='Outliers')
    plt.title(f'Time Series of {column} with Outliers')
    plt.xlabel('Date/Time')
    plt.ylabel(column)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Analyze each numerical column
for column in numerical_columns:
    # Display basic statistics
    print(f"\nStatistics for {column}:")
    print(df[column].describe())
    
    # Detect and print outliers
    outliers = detect_outliers(df, column)
    print(f"\nOutliers in {column}:")
    print(outliers[column])
    
    # Plot time series with outliers
    plot_time_series_with_outliers(df, column)
    
    # Plot histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# Seasonal decomposition for temperature (example)
from statsmodels.tsa.seasonal import seasonal_decompose

# Resample to daily data if needed
daily_data = df['Temp_C'].resample('D').mean()

# Perform seasonal decomposition
result = seasonal_decompose(daily_data, model='additive', period=365)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
result.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
plt.tight_layout()
plt.show()





