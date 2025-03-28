
---

```markdown
# Metro Traffic Volume Prediction

A complete end-to-end machine learning project to predict hourly traffic volume on Metro Interstates. The project includes data preprocessing, feature engineering, model training & evaluation, a FastAPI backend for predictions, a Streamlit frontend for user interaction, and containerization using Docker.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Data Preprocessing](#data-preprocessing)
- [Data Transformation & Model Training](#data-transformation--model-training)
- [API & Frontend](#api--frontend)
- [Usage](#usage)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## Overview

This project predicts hourly traffic volume on a metro interstate using machine learning. The system performs:
- Data ingestion from a Cassandra database.
- Data preprocessing (cleaning, outlier removal, feature extraction, and feature engineering).
- Feature transformation using pipelines (scaling, encoding, and cyclic encoding).
- Model training with multiple machine learning algorithms.
- Evaluation and selection of the best performing model.
- Deployment of a FastAPI backend serving predictions.
- A Streamlit frontend for interactive model testing.
- Containerization with Docker for portability and easy deployment.

## Features

- **Data Preprocessing:** Cleans and prepares raw traffic data.
- **Feature Engineering:** Extracts and transforms time and weather-related features.
- **Model Training:** Evaluates multiple models and selects the best (using CatBoost in this case).
- **API Deployment:** FastAPI-based prediction service.
- **Interactive Frontend:** Streamlit UI for entering traffic conditions and visualizing predictions.
- **Dockerized:** Fully containerized application for easy deployment.

## Technologies

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, scikit-learn, CatBoost, XGBoost, FastAPI, Uvicorn, Streamlit, Plotly
- **Database:** Cassandra (for raw data ingestion)
- **Containerization:** Docker, Docker Compose
- **Version Control:** Git, GitHub

## Project Structure

```
Metro_Traffic_Volume_Prediction/
├── catboost_info/
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv
│   ├── X_processed.csv
│   ├── X_test.csv
│   ├── y_processed.csv
│   └── y_test.csv
├── models/
│   ├── evaluation_results.txt
│   ├── preprocessor.pkl
│   └── traffic_model.pkl
├── notebook/
│   └── Metro_Interstate_Traffic.ipynb
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── data_transformation.py
│   ├── model_evaluation.py
│   └── model_training.py
├── .gitignore
├── README.md
├── __init__.py
├── frontend.py
├── requirements.txt
├── setup.py
├── Dockerfile
├── Dockerfile-frontend
└── docker-compose.yml
```

**Installation & Setup**
### Local Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<your-username>/Metro_Traffic_Volume_Prediction.git
   cd Metro_Traffic_Volume_Prediction
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

4. **Run the Preprocessing, Training, and Evaluation Scripts:**

   - Preprocess the data:
     ```bash
     python src/data_preprocessing.py
     ```
   - Train and evaluate the model:
     ```bash
     python src/model_training.py
     python src/model_evaluation.py
     ```

5. **Run the FastAPI Backend:**

   ```bash
   uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
   ```

6. **Run the Frontend (Streamlit):**

   ```bash
   streamlit run frontend.py
   ```

### Docker Setup

1. **Ensure Docker is Installed:**  
   Follow the [official Docker installation guide](https://docs.docker.com/get-docker/).

2. **Build Docker Images:**

   From the project root, run:
   ```bash
   docker-compose build --no-cache
   ```

3. **Run the Containers:**

   ```bash
   docker-compose up -d
   ```

4. **Access the Applications:**

   - FastAPI Backend: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Streamlit Frontend: [http://localhost:8501](http://localhost:8501)

5. **Stopping the Containers:**

   ```bash
   docker-compose down
   ```

## Data Preprocessing

- The `data_preprocessing.py` script loads data from Cassandra (or CSV for local testing), cleans the data, performs outlier removal on temperature, extracts features (weekday, hour, month, hour_category), and converts the holiday field to binary.
- Preprocessed data is saved to `data/X_processed.csv` and `data/y_processed.csv`.

## Data Transformation & Model Training

- **Transformation:**  
  `data_transformation.py` creates a preprocessing pipeline that scales numerical features, one-hot encodes `weather_main`, ordinal-encodes `hour_category`, passes binary features, and applies cyclic encoding to time features.
  
- **Training:**  
  The `model_training.py` script loads preprocessed data, applies a train-test split, evaluates several models using cross-validation, and selects the best-performing model (CatBoost).  
  It then creates a final pipeline, trains the model, evaluates on the test set, and saves the final model to `models/traffic_model.pkl`.

## API & Frontend

- **FastAPI Backend (`src/app.py`):**  
  Provides an endpoint `/predict` that accepts traffic conditions, preprocesses input data (with helper functions), and returns a traffic volume prediction.
  
- **Streamlit Frontend (`frontend.py`):**  
  An interactive UI for users to input traffic conditions and visualize predictions. The frontend sends requests to the FastAPI backend using the service name in Docker Compose for inter-container communication.

## Usage

1. **Using Local Setup:**  
   - Run the backend and frontend as described in the [Local Setup](#local-setup) section.
   - Test the API at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and interact with the Streamlit app.

2. **Using Docker:**  
   - Start the containers with `docker-compose up -d` and then access the API and frontend using their respective URLs.

## Deployment

For production deployment:
- **CI/CD:** Consider integrating GitHub Actions or Jenkins to automate tests and builds.
- **Cloud Deployment:** Deploy Docker containers to platforms like AWS (ECS, Fargate), Google Cloud Run, or Azure Container Instances.
- **Monitoring:** Set up logging and monitoring (e.g., Prometheus, Grafana) to track app performance and errors.

## Future Work

- **Model Improvements:**  
  Experiment with additional features and further model tuning.
  
- **Scaling:**  
  Scale the solution with Kubernetes and improve the CI/CD pipeline.
  
- **User Feedback:**  
  Integrate user feedback for continuous model improvement.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Developed by **Rohith Obillaneni**  
Email: [your-email@example.com](mailto:your-email@example.com)  
GitHub: [https://github.com/<your-username>](https://github.com/<your-username>)
