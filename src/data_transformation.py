import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Feature transformation functions

# Function for cyclic encoding (for weekday, month)
def cyclic_encode(df):
    df = df.copy()
    for col, max_val in zip(['weekday', 'month','hour'], [7, 12,24]):  # Corrected max values
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df.drop(columns=['weekday', 'month','hour'])


# Data Preprocessing Pipeline (column transformations only)
def get_preprocessing_pipeline():
    # Define feature groups
    num_features = ['temp', 'clouds_all']
    onehot_features = ['weather_main']
    ordinal_features = ['hour_category']
    binary_features = ['holiday']
    cyclic_features = ['weekday', 'month','hour']

    # Create a pipeline for preprocessing
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    onehot_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
    ordinal_transformer = Pipeline(steps=[('encoder', OrdinalEncoder(categories=[['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night']]))])

    # Cyclic encoding transformer
    cyclic_transformer = FunctionTransformer(cyclic_encode, validate=False)

    # Column transformer for preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('onehot', onehot_transformer, onehot_features),
        ('ordinal', ordinal_transformer, ordinal_features),
        ('binary', 'passthrough', binary_features),
        ('cyclic', cyclic_transformer, cyclic_features)
    ])
    print("Data Transformation Completed!")
    return preprocessor
