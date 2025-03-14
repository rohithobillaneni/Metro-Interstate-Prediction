import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Load test data
X_test = pd.read_csv('data/X_processed.csv')
y_test = pd.read_csv('data/y_processed.csv')

# Ensure y is a 1D array
y_test = np.ravel(y_test)

# Load the trained model
with open('models/traffic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f" Model Evaluation Results:\nMAE: {mae:.2f}\nR² Score: {r2:.2f}")

# Optional: Save evaluation results
with open("models/evaluation_results.txt", "w") as f:
    f.write(f"Model Evaluation Results:\nMAE: {mae:.2f}\nR² Score: {r2:.2f}")

print(" Evaluation completed and results saved!")
