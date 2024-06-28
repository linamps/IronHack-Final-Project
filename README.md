## AIRBNB PRICE PREDICTOR

## Git Repository
https://github.com/linamps/IronHack-Final-Project.git

## Tableau Dashboard
https://public.tableau.com/views/AirbnbPriceAnalysis_17193244434260/3_ModelSelection?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link


## Problem Statement
The task is to develop a machine learning model to predict the price of Airbnb listings. The goal is to assist potential hosts in setting competitive prices based on various factors such as location, room type, number of guests, and other amenities.

## Business Case
Accurate pricing is crucial for the success of an Airbnb listing. Setting the right price can:

- Attract more guests.
- Maximize occupancy rates.
- Ensure profitability for hosts.
- This project aims to provide a robust model that predicts Airbnb prices with high accuracy, helping hosts optimize their pricing strategies.

# Project Overview
The project involves:

- Data Collection: Gathering relevant features such as city, room type, number of guests, and additional amenities.
- Data Preprocessing: Cleaning and transforming data for model training.
- Model Selection: Comparing different machine learning algorithms to identify the best-performing model.
- Model Evaluation: Assessing the model's performance using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

Data collection and preprocessing file: data_processing.py
Model Selection and Model Evaluation file: model_selection_training.py

# Technologies Used
- Python (Programming Language)
- Pandas (Data Manipulation)
- NumPy (Numerical Operations)
- Scikit-learn (Machine Learning Library)
- Streamlit (Web Application Framework)
- Matplotlib (Visualization)
- Joblib (Model Serialization)

## Installation
Clone the repository:

bash
git clone https://github.com/linamps/IronHack-Final-Project.git


# Data Description
## Features:

- City: Location of the Airbnb.
- Day: Whether it’s a weekday or weekend.
- Room Type: Type of room (Entire home/apt, Private room, Shared room).
- Shared Room: Number of shared rooms.
- Private Room: Number of private rooms.
- Person Capacity: Maximum number of guests.
- Superhost: Whether the host is a superhost (1 for yes, 0 for no).
- Multiple Rooms: Whether multiple rooms are available (1 for yes, 0 for no).
- Business: Whether the listing is suitable for business (1 for yes, 0 for no).
- Cleanliness Rating: Rating of cleanliness.
- Guest Satisfaction: Guest satisfaction score.
- Bedrooms: Number of bedrooms.
- City Center (km): Distance to the city center in kilometers.
- Metro Distance (km): Distance to the nearest metro station in kilometers.
- Visits: Number of visits to the city.
- Population: Population of the city.
- GDP (bln): GDP of the city in billion dollars.

# Model Selection
## Comparison Summary:

- Linear Regression: High Mean MSE and Low R².
- Decision Tree: Slightly better but less accurate than Random Forest.
- Random Forest: Best performance with the lowest Mean MSE and highest Mean R².
- Gradient Boosting: Competitive but slightly less accurate than Random Forest.
- Support Vector Regressor: Performs worst in terms of MSE and R².
- Extra Trees Regressor: Second-best, with slightly higher Mean MSE than Random Forest.

## Why Random Forest?
Lowest Mean MSE (2920.92).
Highest Mean R² (0.7258).
Consistent Performance (Low Std MSE and Std MAE).

## Results and Evaluation
Random Forest Model:

Mean MSE: 2920.92
Mean MAE: 37.17
Mean R²: 0.7258
Graphical Analysis:


## Conclusions
The Random Forest model was selected due to its superior performance across key metrics, including the lowest Mean MSE, highest Mean R², and consistent error metrics. It provides a robust and reliable model for predicting Airbnb prices.

# Future Work
Model Tuning: Further tuning of Random Forest parameters.
Feature Engineering: Exploring additional features or transformations.
Model Deployment: Consider deploying the model to a cloud platform for real-time predictions.