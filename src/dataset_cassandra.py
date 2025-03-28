from cassandra.cluster import Cluster
import pandas as pd
from cassandra.query import BatchStatement, PreparedStatement

# Connect to Cassandra
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('traffic_data')

# Load CSV
csv_path = r"C:/Users/Rohith/Desktop/Metro_Traffic_Volume_Prediction/data/Metro_Interstate_Traffic_Volume.csv"
df = pd.read_csv(csv_path)

# Handle NaN values
df.fillna({
    'holiday': 'None',
    'weather_main': 'Unknown',
    'weather_description': 'No description'
}, inplace=True)

# Convert date_time column to proper timestamp
df['date_time'] = pd.to_datetime(df['date_time'])

# Prepare insert statement
insert_query = session.prepare("""
    INSERT INTO metro_traffic (date_time, holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, traffic_volume)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""")

# Batch insert
batch = BatchStatement()

for i, row in df.iterrows():
    batch.add(insert_query, (
        row['date_time'].to_pydatetime(),
        row['holiday'],
        float(row['temp']),
        float(row['rain_1h']),
        float(row['snow_1h']),
        int(row['clouds_all']),
        row['weather_main'],
        row['weather_description'],
        int(row['traffic_volume'])
    ))

    if i % 100 == 0:  # Execute every 100 records
        session.execute(batch)
        batch = BatchStatement()

# Execute remaining batch
if batch:
    session.execute(batch)

print("✅ Data uploaded successfully!")
