import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from geopy.geocoders import Nominatim

# --- GLOBAL ARTIFACTS AND CONSTANTS ---

# 1. Initialize the geocoder globally
geolocator = Nominatim(user_agent="us_rental_calculator_app") 

# Define the features that were numerically scaled (StandardScaler/RobustScaler)
# This list MUST match the final numerical columns used in X_train_preprocessed.
# Note: It includes the log-transformed feature names!
SCALING_FEATURES = ["latitude", "longitude", "review_scores_cleanliness", "review_scores_location",
                    "instant_bookable", "log_beds_imputed", "log_accommodates",
                    "log_bedrooms_imputed", "log_bathrooms_imputed"]

# Define the CATEGORICAL feature that was One-Hot Encoded
CATEGORICAL_FEATURE = 'property_category' 

# -----------------------------------------------------
# 2. LOAD THE SAVED ARTIFACTS
# -----------------------------------------------------
@st.cache_resource 
def load_files():
    try:
        # 1. Load Model (best_tree_model.pkl - using pickle)
        with open('best_decisiontree_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # 2. Load Standard Scaler (dt_standardscaler.joblib - using joblib)
        scaler = joblib.load('dt_standardscaler.joblib')
        
        # 3. Load Encoder (dt_encoder.joblib - using joblib)
        encoder = joblib.load('dt_encoder.joblib')

        # 4. Load Column List (dt_columns.json - using json)
        with open('dt_columns.json', 'r') as f:
            model_columns = json.load(f)

        return model, scaler, encoder, model_columns

    except FileNotFoundError as e:
        st.error(f"Error: Could not find required file: {e.filename}. Please ensure all artifacts are in the app's root directory.")
        return None, None, None, None

# Load the four necessary objects
model, scaler, encoder, model_columns = load_files()


if model is not None:
    # --- 3. TITLE AND UI LAYOUT ---
    st.title("🏡 Airbnb Price Predictor")
    st.write("Enter your listing details and quality scores below.")
    st.markdown("---")

    # Layout: Use columns for a cleaner UI
    col1, col2 = st.columns(2)
    
    # --- INPUTS ---
    with col1:
        st.subheader("Physical Properties")
        # Raw inputs (will be log-transformed later)
        accommodates = st.number_input("1. Accommodates (Guests)", min_value=1, value=2)
        bedrooms = st.number_input("2. Bedrooms", min_value=0, value=1)
        beds = st.number_input("3. Beds", min_value=1, value=1)
        bathrooms = st.number_input("4. Bathrooms", min_value=0.5, value=1.0)
        
        st.subheader("Location")
        address_input = st.text_input("5. Enter US Address or City Name", 
                                     placeholder="e.g., San Francisco, CA",
                                     key='address_input')
        
    with col2:
        st.subheader("Quality & Booking")
        # Raw inputs (no log-transform needed)
        review_scores_cleanliness = st.slider("6. Cleanliness Score (1-5)", 1.0, 5.0, 4.5)
        review_scores_location = st.slider("7. Location Score (1-5)", 1.0, 5.0, 4.5)
        instant_bookable = st.selectbox("8. Instant Bookable", ["Yes", "No"])
        instant_bookable_nb = 1 if instant_bookable == "Yes" else 0 # Encode to 1 or 0
        
        # Categorical Input for Property Category (One-Hot Encoded)
        property_categories = [
            'Entire Apartment/Condo', 'Entire House', 'Private Room', 
            'Hotel/Resort'
        ]
        property_category = st.selectbox("9. Property Type Category", property_categories)

    st.markdown("---")
    
    # --- 4. PREDICTION LOGIC ---
    if st.button("Predict Price 💰", use_container_width=True):
        
        # 4A. GEOCODING STEP
        if not address_input:
            st.error("Please enter a valid US address or city name to continue.")
            st.stop()
            
        with st.spinner('Searching for coordinates...'):
            try:
                location = geolocator.geocode(address_input, timeout=10)
                
                if location is None:
                    st.error("Could not find coordinates for that address. Please be more specific (e.g., City, State).")
                    st.stop()
                    
                latitude = location.latitude
                longitude = location.longitude
                st.info(f"Location found: Lat={latitude:.4f}, Lon={longitude:.4f}")
                
            except Exception as e:
                st.error("Geocoding service failed. Check your internet connection or try a different address format.")
                st.exception(e)
                st.stop()
        
        # B. Create Raw Input DataFrame and Apply Log Transforms
        # NOTE: Using np.log() for the four specified features
        raw_input_data = pd.DataFrame({
            # Apply Log Transformation
            "log_accommodates": [np.log(accommodates)],
            "log_beds_imputed": [np.log(beds)],
            "log_bedrooms_imputed": [np.log(bedrooms)],
            "log_bathrooms_imputed": [np.log(bathrooms)],
            
            # Remaining Numerical Features (No log)
            "latitude": [latitude], 
            "longitude": [longitude], 
            "review_scores_cleanliness": [review_scores_cleanliness],
            "review_scores_location": [review_scores_location],
            "instant_bookable": [instant_bookable_nb],
            
            # Raw Categorical Feature
            CATEGORICAL_FEATURE: [property_category]
        })

        # C. Apply Preprocessing (Standard Scaling & One-Hot Encoding)

        # 1. Apply Standard Scaler to all SCALING_FEATURES
        raw_input_data[SCALING_FEATURES] = scaler.transform(raw_input_data[SCALING_FEATURES])
        
        # 2. Apply One-Hot Encoder to the categorical column
        ohe_output = encoder.transform(raw_input_data[[CATEGORICAL_FEATURE]])
        
        # Get the new column names from the encoder
        ohe_col_names = encoder.get_feature_names_out([CATEGORICAL_FEATURE])
        ohe_df = pd.DataFrame(ohe_output, columns=ohe_col_names)
        
        # 3. Concatenate all features (Scaled Numerical + OHE)
        features_to_keep = [c for c in raw_input_data.columns if c not in [CATEGORICAL_FEATURE]]
        final_processed_df = pd.concat([raw_input_data[features_to_keep].reset_index(drop=True), ohe_df], axis=1)

        # D. FINAL STEP: Reorder and Align Columns
        # Use model_columns (the saved list) to ensure correct order
        input_data = final_processed_df.reindex(columns=model_columns, fill_value=0)

        # E. Predict (The model predicts the log price)
        try:
            log_prediction = model.predict(input_data)[0]
            
            # F. Convert back to Dollars (Inverse Transformation)
            price_prediction = np.exp(log_prediction)
            
            st.success(f"### Estimated Price: **${price_prediction:,.2f}** per night")
            st.balloons()
            
        except Exception as e:
            st.error("Prediction failed. Ensure all features are correctly formatted and in the correct order.")
            st.exception(e)