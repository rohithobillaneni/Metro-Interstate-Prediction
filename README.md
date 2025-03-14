# Traffic Volume Prediction Project

This project predicts **traffic volume** based on **weather conditions, holidays, and time-based features**.

## Features:
- **Weather conditions** (rain, snow, clouds, temperature)
- **Holiday effects**
- **Time-based trends**

## How to Run:
1. Install dependencies:

pip install -r requirements.txt

2. Run preprocessing:
python src/data_preprocessing.py

3. Train the model:
python src/model_training.py

4. Evaluate model:
python src/model_evaluation.py

5. Run API:
python deployment/app.py


## API Usage:
- Endpoint: `POST /predict`
- Input JSON:  
```json
{"temp": 15, "rain_1h": 0, "snow_1h": 0, "clouds_all": 50, "weather_main": "Clear", "holiday": "None"}