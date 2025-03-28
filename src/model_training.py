import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from data_preprocessing import load_and_preprocess_data
from data_transformation import get_preprocessing_pipeline

# Step 1: Load and preprocess the data (from data_preprocessing.py)
X, y = load_and_preprocess_data()  

# Step 2: Get the preprocessing pipeline (from data_transformation.py)
preprocessor = get_preprocessing_pipeline()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure y is a 1D array
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)

# Save to CSV
X_test_df.to_csv('data/X_test.csv', index=False)
y_test_df.to_csv('data/y_test.csv', index=False)

# Check data consistency
assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train samples"
assert X_test.shape[0] == y_test.shape[0], "Mismatch in X_test and y_test samples"

# List of models to evaluate
models = [
    ('RandomForest', RandomForestRegressor()),
    ('GradientBoost', GradientBoostingRegressor()),
    ('AdaBoost', AdaBoostRegressor()),
    ('CatBoost', CatBoostRegressor(verbose=0)),
    ('XGBoost', XGBRegressor())
]

# Evaluate models using cross-validation
r2_scores = []
for name, model in models:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    r2 = cross_val_score(pipeline, X_train, y_train, cv=KFold(n_splits=10), scoring='r2', n_jobs=-1).mean()
    r2_scores.append((name, r2))

# Convert results to DataFrame
results_df = pd.DataFrame(r2_scores, columns=['Model', 'R² Score']).sort_values(by='R² Score', ascending=False)

# Print model performances
print(results_df)

# Plot model performance
import seaborn as sns
import matplotlib.pyplot as plt
1
plt.figure(figsize=(10, 5))
sns.barplot(y=results_df['Model'], x=results_df['R² Score'])
plt.title('Model Comparison (R² Score)')
plt.show()

# Select the best model (CatBoost) for final training
best_model = CatBoostRegressor(iterations=100, depth=8, learning_rate=0.1, loss_function='RMSE', random_seed=42, bagging_temperature=0.95, min_data_in_leaf=500)

# Create final pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Train the final model
final_pipeline.fit(X_train, y_train)

# Evaluate the final model on the test data
y_pred = final_pipeline.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Final Model Evaluation Results:\nMAE: {mae:.2f}\nR² Score: {r2:.2f}")

# Save trained model
with open('models/traffic_model.pkl', 'wb') as model_file:
    pickle.dump(final_pipeline, model_file)

# Optionally, save evaluation results
with open("models/evaluation_results.txt", "w") as f:
    f.write(f"Final Model Evaluation Results:\nMAE: {mae:.2f}\nR² Score: {r2:.2f}")

print("🎉 Model Training & Saving Completed!")
