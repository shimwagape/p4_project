import pandas as pd
import streamlit as st
import numpy as np
import pickle
# streamlit import sklearn
from PIL import Image
import os
import xgboost as xgb

# model = xgb.XGBRegressor()
# model = model.load_model('export/xg_model.json')

# Load the saved components:
with open(".\export\dt_model.pkl", "rb") as f:
    components = pickle.load(f)

# Extract the individual components
num_imputer = components["num_imputer"]
cat_imputer = components["cat_imputer"]
encoder = components["encoder"]
scaler = components["scaler"]
dt_model = components["models"]

# Create the app

st.set_page_config(
    layout="wide"
)


# # Add an image or logo to the app
# image = Image.open('copofav.jpg')

# # Open the image file
# st.image(image)


# add app title
st.title("SALES PREDICTION APP")


# Add some text
st.write("Please ENTER the relevant data and CLICK Predict.")

# Create the input fields
input_data = {}
col1, col2, col3 = st.columns(3)
with col1:
    input_data['store_nbr'] = st.slider(
        "Store Number", min_value=0, step=1, max_value=54)
    input_data['family'] = st.selectbox("Products Family", ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
                                                            'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
                                                            'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
                                                            'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
                                                            'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
                                                            'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
                                                            'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
                                                            'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
                                                            'SEAFOOD'])
    input_data['onpromotion'] = st.number_input(
        "Discount Amt On Promotion", step=1)

with col2:
    input_data['state'] = st.selectbox("State", ['Santa Elena', 'Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
                                                 'Santo Domingo de los Tsachilas', 'Bolivar', 'Tungurahua',
                                                 'Guayas', 'Los Rios', 'Azuay', 'Loja', 'El Oro', 'Esmeraldas',
                                                 'Manabi', 'Pastaza'])
    input_data['store_type'] = st.radio(
        "Store Type", options=['A', 'B', 'C', 'D', 'E'], horizontal=True)
    input_data['cluster'] = st.number_input("Cluster", step=1)

with col3:
    input_data['month'] = st.slider("Month", 1, 12)
    input_data['day'] = st.slider("Day", 1, 31)
    input_data['dcoilwtico'] = st.slider(
        "DCOILWTICO", min_value=29, step=1, max_value=108)

  # Create a button to make a prediction
if st.button("Predict"):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # # categorizing the products
    # food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI','PRODUCE', 'DAIRY','POULTRY','EGGS','SEAFOOD']
    # home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
    # clothing_families = ['LINGERIE', 'LADYSWARE']
    # grocery_families = ['GROCERY I', 'GROCERY II']
    # stationery_families = ['BOOKS', 'MAGAZINES','SCHOOL AND OFFICE SUPPLIES']
    # cleaning_families = ['HOME CARE', 'BABY CARE','PERSONAL CARE']
    # hardware_families = ['PLAYERS AND ELECTRONICS','HARDWARE']
    # others_families = ['AUTOMOTIVE', 'BEAUTY','CELEBRATION', 'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',  'PET SUPPLIES']

    # # Apply the same preprocessing steps as done during training
    # input_df['products'] = np.where(input_df['products'].isin(food_families), 'FOODS', input_df['products'])
    # input_df['products'] = np.where(input_df['products'].isin(home_families), 'HOME', input_df['products'])
    # input_df['products'] = np.where(input_df['products'].isin(clothing_families), 'CLOTHING', input_df['products'])
    # input_df['products'] = np.where(input_df['products'].isin(grocery_families), 'GROCERY', input_df['products'])
    # input_df['products'] = np.where(input_df['products'].isin(stationery_families), 'STATIONERY', input_df['products'])
    # input_df['products'] = np.where(input_df['products'].isin(cleaning_families), 'CLEANING', input_df['products'])
    # input_df['products'] = np.where(input_df['products'].isin(hardware_families), 'HARDWARE', input_df['products'])
    # input_df['products'] = np.where(input_df['products'].isin(others_families), 'OTHERS', input_df['products'])

    categorical_columns = ['family', 'state', 'store_type']
    numerical_columns = ['store_nbr', 'onpromotion',
                         'cluster', 'dcoilwtico', 'month', 'day']
    # Impute missing values
    input_df_cat = input_df[categorical_columns].copy()
    input_df_num = input_df[numerical_columns].copy()
    input_df_cat_imputed = cat_imputer.fit_transform(input_df_cat)
    input_df_num_imputed = num_imputer.fit_transform(input_df_num)

    # Encode categorical features
    input_df_cat_encoded = encoder.fit(input_df_cat_imputed)
    input_df_cat_encoded = pd.DataFrame(encoder.transform(input_df_cat_imputed).toarray(),
                                        columns=encoder.get_feature_names_out(categorical_columns))

    # Scale numerical features
    input_df_num_scaled = scaler.fit_transform(input_df_num_imputed)
    input_df_num_sc = pd.DataFrame(
        input_df_num_scaled, columns=numerical_columns)

    # Combine encoded categorical features and scaled numerical features
    input_df_processed = pd.concat(
        [input_df_num_sc, input_df_cat_encoded], axis=1)

    # Make predictions using the trained model
    predictions = dt_model.predict(input_df_processed)
    # predictions = model.predict(input_df_processed)

    # Display the predicted sales value to the user:
    st.write("Predicted Sales:", predictions[0])
