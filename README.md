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

Metro_Traffic_Volume_Prediction/ ├── catboost_info/ ├── data/ │ ├── Metro_Interstate_Traffic_Volume.csv │ ├── X_processed.csv │ ├── X_test.csv │ ├── y_processed.csv │ └── y_test.csv ├── models/ │ ├── evaluation_results.txt │ ├── preprocessor.pkl │ └── traffic_model.pkl ├── notebook/ │ └── Metro_Interstate_Traffic.ipynb ├── src/ │ ├── init.py │ ├── app.py │ ├── data_preprocessing.py │ ├── data_transformation.py │ ├── model_evaluation.py │ └── model_training.py ├── .gitignore ├── README.md ├── init.py ├── frontend.py ├── requirements.txt ├── setup.py ├── Dockerfile ├── Dockerfile-frontend └── docker-compose.yml


## Installation & Setup

### Local Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<your-username>/Metro_Traffic_Volume_Prediction.git
   cd Metro_Traffic_Volume_Prediction


2. **Create and Activate a Virtual Environment:**

    '''bash
    python -m venv venv
    source venv/bin/activate      # On Windows: venv\Scripts\activate

3. **Install Dependencies:**

    '''bash
    pip install --no-cache-dir -r requirements.txt

4. **Run the Preprocessing, Training, and Evaluation Scripts:**

    Preprocess the data:

    '''bash
    python src/data_preprocessing.py

    Train and evaluate the model:

    '''bash
    python src/model_training.py
    python src/model_evaluation.py

    Run the FastAPI Backend:
    '''bash
    uvicorn src.app:app --reload --host 127.0.0.1 --port 8000

    Run the Frontend (Streamlit):

    '''bash
    streamlit run frontend.py


**Docker Setup**

1. **Ensure Docker is Installed:**
    Follow the official Docker installation guide.

2. **Build Docker Images:**
    From the project root, run:

    '''bash
    docker-compose build --no-cache

3. **Run the Containers:**

    '''bash
    docker-compose up -d

4. **Access the Applications:**

    FastAPI Backend: http://localhost:8000/docs

    Streamlit Frontend: http://localhost:8501

5. **Stopping the Containers:**

    '''bash
    docker-compose down
