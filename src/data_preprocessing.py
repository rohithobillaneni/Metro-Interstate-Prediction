import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')

# Convert date_time to datetime format
df['date_time'] = pd.to_datetime(df['date_time'])

# Convert temperature from Kelvin to Celsius
df['temp'] = df['temp'] - 273.15

# Remove unnecessary columns
df.drop(columns=['weather_description', 'snow_1h', 'rain_1h'], inplace=True)

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'temp')  # Removing outliers in temperature

# Extract new features from date_time
df['weekday'] = df['date_time'].dt.weekday
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month

# Categorize hours into time slots
def categorize_hour(hour):
    if hour in [4, 5, 6, 7]:
        return 'Early Morning'
    elif hour in [8, 9, 10, 11]:
        return 'Morning'
    elif hour in [12, 13, 14, 15]:
        return 'Afternoon'
    elif hour in [16, 17, 18, 19]:
        return 'Evening'
    elif hour in [20, 21, 22, 23]:
        return 'Night'
    else:
        return 'Late Night'

df['hour_category'] = df['hour'].apply(categorize_hour)

# Convert holiday column to binary (0 = No Holiday, 1 = Holiday)
df['holiday'] = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)

# Drop original date_time column
df.drop(columns=['date_time'], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['weather_main'] = le.fit_transform(df['weather_main'])
df['hour_category'] = le.fit_transform(df['hour_category'])

# Define features and target
X = df.drop(columns=['traffic_volume'])
y = df['traffic_volume']

# Save preprocessed data
X.to_csv('data/X_processed.csv', index=False)
y.to_csv('data/y_processed.csv', index=False)

# Save encoders
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))

print(" Data Preprocessing Completed!")
