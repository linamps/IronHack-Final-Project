import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'airbnb_price_predictor2.pkl')
model = joblib.load(model_path)

# Define a dictionary mapping cities to their respective data
city_info = {
    'Amsterdam': {'visits': '8,000,000', 'Population': '872,000', 'GDP(bln)': '320'},
    'Athens': {'visits': '5,500,000', 'Population': '664,000', 'GDP(bln)': '130'},
    'Barcelona': {'visits': '9,000,000', 'Population': '1,620,000', 'GDP(bln)': '200'},
    'Berlin': {'visits': '13,000,000', 'Population': '3,570,000', 'GDP(bln)': '160'},
    'Budapest': {'visits': '4,000,000', 'Population': '1,750,000', 'GDP(bln)': '140'},
    'Lisbon': {'visits': '4,500,000', 'Population': '545,000', 'GDP(bln)': '110'},
    'Paris': {'visits': '19,000,000', 'Population': '2,160,000', 'GDP(bln)': '900'},
    'Rome': {'visits': '10,000,000', 'Population': '2,870,000', 'GDP(bln)': '200'},
    'Vienna': {'visits': '7,000,000', 'Population': '1,930,000', 'GDP(bln)': '120'}
}

# Define a function to preprocess input data
def preprocess_data(input_data):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Perform any necessary preprocessing steps here
    city_mapping = {'Amsterdam': 0, 'Athens': 1, 'Barcelona': 2, 'Berlin': 3, 'Budapest': 4, 'Lisbon': 5, 'Paris': 6, 'Rome': 7, 'Vienna': 8}
    day_mapping = {'Weekday': 0, 'Weekend': 1}
    room_mapping = {'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2}
    
    # Create mappings for visits and population
    visits_mapping = {'8,000,000': 0, '5,500,000': 1, '9,000,000': 2, '13,000,000': 3, '4,000,000': 4, '4,500,000': 5, '19,000,000': 6, '10,000,000': 7, '7,000,000': 8}
    population_mapping = {'872,000': 0, '664,000': 1, '1,620,000': 2, '3,570,000': 3, '1,750,000': 4, '545,000': 5, '2,160,000': 6, '2,870,000': 7, '1,930,000': 8}
    gdp_mapping = {'320': 0, '130': 1, '200': 2, '160': 3, '140': 4, '110': 5, '900': 6, '120': 7}
    
    input_df['City'] = input_df['City'].map(city_mapping)
    input_df['Day'] = input_df['Day'].map(day_mapping)
    input_df['Room Type'] = input_df['Room Type'].map(room_mapping)
    input_df['visits'] = input_df['visits'].map(visits_mapping)
    input_df['Population'] = input_df['Population'].map(population_mapping)
    input_df['GDP(bln)'] = input_df['GDP(bln)'].map(gdp_mapping)
    
    # Convert visits and population to categorical variables
    input_df['visits'] = pd.Categorical(input_df['visits'], categories=[0, 1, 2, 3, 4, 5, 6, 7, 8], ordered=True)
    input_df['Population'] = pd.Categorical(input_df['Population'], categories=[0, 1, 2, 3, 4, 5, 6, 7, 8], ordered=True)
    
    # Create dummy variables for categorical columns
    input_df = pd.get_dummies(input_df, columns=['City', 'Day', 'Room Type', 'visits', 'Population'])
    
    # Ensure all features are present and in the correct order
    expected_columns = ['City_0', 'City_1', 'City_2', 'City_3', 'City_4', 'City_5', 'City_6', 'City_7', 'City_8',
                        'Day_0', 'Day_1', 'Room Type_0', 'Room Type_1', 'Room Type_2',
                        'visits_0', 'visits_1', 'visits_2', 'visits_3', 'visits_4', 'visits_5', 'visits_6', 'visits_7', 'visits_8',
                        'Population_0', 'Population_1', 'Population_2', 'Population_3', 'Population_4', 'Population_5', 'Population_6', 'Population_7', 'Population_8',
                        'Shared Room', 'Private Room', 'Person Capacity', 'Superhost', 'Multiple Rooms',
                        'Business', 'Guest Satisfaction', 'Bedrooms', 'City Center (km)',
                        'Metro Distance (km)', 'GDP(bln)']
    
    # Reindex input_df to match expected columns and fill missing columns with zeros
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    return input_df

# Streamlit app
st.title("Airbnb Price Prediction")


st.write("""
### Enter the details of your Airbnb property to get a price suggestion
""")

# Create input fields for each feature
city = st.selectbox('City', list(city_info.keys()))
day = st.selectbox('Day', ['Weekday', 'Weekend'])
room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Shared room'])
shared_room = st.number_input('Shared Room', value=0)
private_room = st.number_input('Private Room', value=0)
person_capacity = st.number_input('Person Capacity', value=1, step=1)
superhost = st.number_input('Superhost', value=0)
multiple_rooms = st.number_input('Multiple Rooms', value=0, step=1)
business = st.number_input('Business', value=0, step=1)
guest_satisfaction = st.number_input('Guest Satisfaction', value=10, step=10)
bedrooms = st.number_input('Bedrooms', value=0, step=1)
city_center = st.number_input('City Center (km)', value=0, step=1)
metro = st.number_input('Metro Distance (km)', value=0, step=1)

# Display visits, population, and GDP based on selected city
if city:
    st.write(f"**Visits:** {city_info[city]['visits']}")
    st.write(f"**Population:** {city_info[city]['Population']}")
    st.write(f"**GDP (bln):** {city_info[city]['GDP(bln)']}")

# Automatically fill in visits, population, and GDP fields
visits = city_info[city]['visits']
population = city_info[city]['Population']
gdp = city_info[city]['GDP(bln)']

# Button to trigger prediction
if st.button('Predict Price'):
    # Prepare input data
    input_data = {
        'City': city,
        'Day': day,
        'Room Type': room_type,
        'Shared Room': shared_room,
        'Private Room': private_room,
        'Person Capacity': person_capacity,
        'Superhost': superhost,
        'Multiple Rooms': multiple_rooms,
        'Business': business,
        'Guest Satisfaction': guest_satisfaction,
        'Bedrooms': bedrooms,
        'City Center (km)': city_center,
        'Metro Distance (km)': metro,
        'visits': visits,
        'Population': population,
        'GDP(bln)': gdp
    }

    # Preprocess input data
    input_df = preprocess_data(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_df)

    # Display prediction
    st.subheader(f"Predicted Price: EUR {prediction[0]:.2f}")
