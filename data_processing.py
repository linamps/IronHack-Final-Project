import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Aemf1.csv') # airbnb data

gdp_df = pd.read_csv('gdp_tourism.csv') # GDP and tourism data

gdp_df = gdp_df.iloc[:,:-5] # excluding last 5 columns

gdp_df = gdp_df.rename(columns={'city': 'City'}) # renaming 'City' column to match for merging

# merging tables
merged_df = pd.merge(df, gdp_df, on='City', how='left')

# changing bool columns to int
merged_df['Shared Room'] = merged_df['Shared Room'].astype(int)
merged_df['Private Room'] = merged_df['Private Room'].astype(int)
merged_df['Superhost'] = merged_df['Superhost'].astype(int)

# Excluding possible outliers

outliers = ['Price', 'City Center (km)', 'Metro Distance (km)', 'Attraction Index', 'Normalised Attraction Index', 'Restraunt Index', 'Normalised Restraunt Index'] 

for col in outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    limit_low = Q1 - IQR * 1.7
    limit_high = Q3 + IQR * 1.7
    
    merged_df = merged_df[(merged_df[col] >= limit_low) & (merged_df[col] <= limit_high)]


# passing object columns to numeric

list_str = merged_df.select_dtypes(include = 'object').columns
le = LabelEncoder()

for c in list_str:
    merged_df[c] = le.fit_transform(merged_df[c])


merged_df.to_csv('airbnb_merged_df.csv', index=False)