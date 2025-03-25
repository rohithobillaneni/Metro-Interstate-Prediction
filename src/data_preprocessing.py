import pandas as pd
import numpy as np

# Load and preprocess data
def load_and_preprocess_data(filepath='data/Metro_Interstate_Traffic_Volume.csv'):
    df = pd.read_csv(filepath)

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
    #df.drop(columns=['hour'], inplace=True)

    # Convert holiday column to binary (0 = No Holiday, 1 = Holiday)
    df['holiday'] = df['holiday'].notna().astype(int)

    # Set the date_time as index
    df.set_index('date_time', inplace=True)

    # Define features and target
    X = df.drop(columns=['traffic_volume'])
    y = df['traffic_volume']

    # Save preprocessed data for training
    X.to_csv('data/X_processed.csv', index=False)
    y.to_csv('data/y_processed.csv', index=False)

    print("Data Preprocessing Completed and Data Saved!")

    return X, y