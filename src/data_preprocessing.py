import pandas as pd
import numpy as np
from cassandra.cluster import Cluster

# Function to load and preprocess data from Cassandra
def load_and_preprocess_data():
    print("🔹 Connecting to Cassandra Database...")

    # Connect to Cassandra
    cluster = Cluster(['127.0.0.1']) 
    session = cluster.connect('traffic_data')  # Connect to the keyspace
    print("✅ Connected to Cassandra.")

    # Query to fetch data
    query = "SELECT date_time, holiday, temp, clouds_all, weather_main, traffic_volume FROM metro_traffic"
    rows = session.execute(query)

    print(f"🔹 Fetched {len(rows.current_rows)} records from Cassandra.")

    # Convert fetched data to a Pandas DataFrame
    df = pd.DataFrame(rows, columns=['date_time', 'holiday', 'temp', 'clouds_all', 'weather_main', 'traffic_volume'])


    # Convert date_time to datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])


    # Convert temperature from Kelvin to Celsius
    df['temp'] = df['temp'] - 273.15

    # Function to remove outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"🔹 Removed outliers from {column}. Original size: {len(df)}, New size: {len(df_filtered)}")
        return df_filtered

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
    df['holiday'] = df['holiday'].apply(lambda x: 0 if x is None else 1)


    # Set the date_time as index
    df.set_index('date_time', inplace=True)


    # Define features and target
    X = df.drop(columns=['traffic_volume'])
    y = df['traffic_volume']

    # Save preprocessed data for training
    X.to_csv('data/X_processed.csv', index=False)
    y.to_csv('data/y_processed.csv', index=False)

    print("Data Preprocessing Completed Successfully!")

    return X, y

