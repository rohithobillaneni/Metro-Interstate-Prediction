import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score

# Load preprocessed data
X = pd.read_csv('data/X_processed.csv')
y = pd.read_csv('data/y_processed.csv')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np

# Ensure y is a 1D array
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Check data consistency
assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train samples"
assert X_test.shape[0] == y_test.shape[0], "Mismatch in X_test and y_test samples"

# Define numerical and categorical features
num_features = ['temp', 'clouds_all']
cat_features = ['holiday', 'weather_main', 'weekday', 'hour_category', 'month']

# Create preprocessing pipeline
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('encoder', OrdinalEncoder())])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

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
plt.figure(figsize=(10, 5))
sns.barplot(y=results_df['Model'], x=results_df['R² Score'])
plt.title('Model Comparison (R² Score)')
plt.show()

# Select the best model (CatBoost)
best_model = CatBoostRegressor(iterations=100, depth=8, learning_rate=0.1, loss_function='RMSE', random_seed=42, bagging_temperature=0.95, min_data_in_leaf=500)

# Create final pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Train the final model
final_pipeline.fit(X_train, y_train)

# Save trained model
pickle.dump(final_pipeline, open('models/traffic_model.pkl', 'wb'))

print("🎉 Model Training & Saving Completed!")
