import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data
from data_transformation import get_preprocessing_pipeline

# Step 1: Load the trained model
with open('models/traffic_model.pkl', 'rb') as model_file:
    final_pipeline = pickle.load(model_file)

# Step 2: Load and preprocess the data (same as in training)
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Step 4: Make predictions on the test set
y_pred = final_pipeline.predict(X_test)

# Step 5: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 6: Print the evaluation metrics
print(f"Model Evaluation Results:\n")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 7: Optionally, save the evaluation results
with open("models/evaluation_results.txt", "w") as f:
    f.write(f"Model Evaluation Results:\nMAE: {mae:.2f}\nR² Score: {r2:.2f}")

print("🎉 Model Evaluation Completed!")
