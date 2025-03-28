from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI()


# Use an absolute path inside the container for the model file.
MODEL_PATH = os.path.join("/app", "models", "traffic_model.pkl")

# Use below path if running locally
#MODEL_PATH = 'C:/Users/Rohith/Desktop/Metro_Traffic_Volume_Prediction/models/traffic_model.pkl'

print(f"Looking for model at: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as model_file:
    final_pipeline = pickle.load(model_file)

    
# Input data schema using Pydantic for validation
class TrafficData(BaseModel):
    temp: float
    clouds_all: int
    holiday: str
    weather_main: str
    weekday: str
    hour: int
    hour_category: str
    month: str

# Helper function for encoding inputs
def preprocess_input(data: dict):
    # Define mappings for categorical features
    encoding_map = {
        "holiday": {"No": 0, "Yes": 1},
        "weekday": {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6},
        "month": {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                  "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    }

    # Check if any required columns are missing
    missing_columns = [col for col in encoding_map.keys() if col not in data]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_columns)}")

    # Map categorical columns to integers
    for col, mapping in encoding_map.items():
        if col in data:
            if isinstance(data[col], str):  # Ensure categorical values are strings
                data[col] = mapping.get(data[col], data[col])  # Use the mapping, or keep the same if not found
            else:
                raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {data[col]}. Expected a string value.")

    # Ensure numeric columns are properly handled
    if not isinstance(data['temp'], (int, float)) or not isinstance(data['clouds_all'], (int, float)):
        raise HTTPException(status_code=400, detail=f"Invalid value for numeric fields. Expected numbers.")

    # Convert the processed data back to a dictionary
    return data


# Prediction endpoint
@app.post("/predict")
async def predict_traffic(data: TrafficData):
    try:
        # Preprocess input data
        data_dict = data.dict()

        # Check that all values are valid
        for key, value in data_dict.items():
            # Ensure numeric fields are actually numbers
            if key in ["temp", "clouds_all", 'hour']:
                if not isinstance(value, (int, float)):
                    raise HTTPException(status_code=400, detail=f"Invalid value for {key}: {value}. Must be a numeric type.")
            
            # Ensure categorical fields are strings and check if they exist in the encoding map
            if key in ["holiday", "weather_main", "weekday", "hour_category", "month"]:
                if not isinstance(value, str):
                    raise HTTPException(status_code=400, detail=f"Invalid value for {key}: {value}. Must be a string type.")
        
        # Process the input data
        processed_data = preprocess_input(data_dict)

        # Convert the processed data into a DataFrame for prediction
        input_data = pd.DataFrame([processed_data])

        print("Received data:")
        print(input_data)

        # Use the final pipeline (which includes both the preprocessor and model)
        prediction = final_pipeline.predict(input_data)[0]
        
        # Return the prediction
        return {"prediction": prediction}

    except HTTPException as e:
        raise e  # Propagate HTTPExceptions for invalid data types
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint (optional, for basic check)
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Metro Traffic Volume Prediction API!"}
