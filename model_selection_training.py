import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import joblib
import os

# Load processed and cleaned data
df = pd.read_csv('merged_df_before_encoding.csv')

# Define feature columns and target
X = df.drop(['Price','Attraction Index','Restraunt Index','Cleanliness Rating', 'Normalised Attraction Index', 'Normalised Restraunt Index','Unnamed: 0'], axis = 1)
y = df['Price']

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Define categorical and numerical columns
categorical_features = ['City', 'Day', 'Room Type', 'visits', 'Population']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Models to be trained
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'Extra Trees Regressor': ExtraTreesRegressor(random_state=42)
}

# Model selection
results = {}
for name, model in models.items():
    # Create a pipeline with the preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # Perform cross-validation for MSE
    mse_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -mse_cv_scores

    # Perform cross-validation for MAE
    mae_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mae_scores = -mae_cv_scores

    # Perform cross-validation for R²
    r2_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')

    results[name] = {
        'Mean MSE': np.mean(mse_scores),
        'Std MSE': np.std(mse_scores),
        'Mean MAE': np.mean(mae_scores),
        'Std MAE': np.std(mae_scores),
        'Mean R²': np.mean(r2_cv_scores),
        'Std R²': np.std(r2_cv_scores)
    }
    print(f"{name}: Mean MSE = {np.mean(mse_scores)}, Std MSE = {np.std(mse_scores)}, "
          f"Mean MAE = {np.mean(mae_scores)}, Std MAE = {np.std(mae_scores)}, "
          f"Mean R² = {np.mean(r2_cv_scores)}, Std R² = {np.std(r2_cv_scores)}")

# Find the best model based on the lowest Mean MSE
best_model_name = min(results, key=lambda x: results[x]['Mean MSE'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Create a pipeline with the best model
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])

# Train the best model
best_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = best_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test R²: {r2}")

# Create and print model summary
summary_df = pd.DataFrame(results).T
models_summary = pd.DataFrame(summary_df)
print(models_summary)


# Save model
import joblib

# Save the model to a file
joblib.dump(best_model, 'airbnb_price_predictor2.pkl')