import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from geopy.geocoders import Nominatim

# --- 1. CONFIGURATION & CONSTANTS ---

# Initialize Geocoder
geolocator = Nominatim(user_agent="us_rental_calculator_app") 

# [CRITICAL] THE EXACT LIST OF 30 TRAINING CITIES

CITY_MAPPING = {
    # CLEAN DISPLAY NAME: EXACT TRAINED VALUE (Must match original DF)
    "Washington DC": "Washington Dc, DC",  # <--- FIX APPLIED HERE
    "New York City": "New York City, NY",
    "Los Angeles, CA": "Los Angeles, CA",
    "San Francisco, CA": "San Francisco, CA",
    "Las Vegas (Clark Co.)": "Clark County, NV", # Example of another clean up
    "Rhode Island, RI": "Rhode Island, RI",
    "Denver, CO": "Denver, CO",
    "Broward County, FL": "Broward County, FL",
    "Hawaii, HI": "Hawaii, HI",
    "Dallas, TX": "Dallas, TX",
    "Santa Clara County, CA": "Santa Clara County, CA",
    "Rochester, NY": "Rochester, NY",
    "Seattle, WA": "Seattle, WA",
    "Austin, TX": "Austin, TX",
    "Twin Cities, MN (MSA)": "Twin Cities MSA, MN",
    "New Orleans, LA": "New Orleans, LA",
    "Fort Worth, TX": "Fort Worth, TX",
    "Boston, MA": "Boston, MA",
    "San Mateo County, CA": "San Mateo County, CA",
    "Jersey City, NJ": "Jersey City, NJ",
    "San Diego, CA": "San Diego, CA",
    "Cambridge, MA": "Cambridge, MA",
    "Oakland, CA": "Oakland, CA",
    "Nashville, TN": "Nashville, TN",
    "Portland, OR": "Portland, OR",
    "Columbus, OH": "Columbus, OH",
    "Santa Cruz County, CA": "Santa Cruz County, CA",
    "Asheville, NC": "Asheville, NC",
    "Pacific Grove, CA": "Pacific Grove, CA",
    "Bozeman, MT": "Bozeman, MT"
}

ALLOWED_CITIES_DISPLAY = list(CITY_MAPPING.keys())

# Features that need Standard Scaling
SCALING_FEATURES = [
    "latitude", "longitude", 
    "review_scores_cleanliness", "review_scores_location",
    "instant_bookable", 
    "log_beds_imputed", "log_accommodates",
    "log_bedrooms_imputed", "log_bathrooms_imputed"
]

# Categorical Feature for One-Hot Encoding
CATEGORICAL_FEATURE = 'property_category' 

# --- 2. LOAD SAVED ARTIFACTS ---
@st.cache_resource 
def load_files():
    try:
        # Load Model
        with open('best_decisiontree_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Load Scaler & Encoder (using joblib)
        scaler = joblib.load('dt_standardscaler.joblib')
        encoder = joblib.load('dt_encoder.joblib')

        # Load Column List
        with open('dt_columns.json', 'r') as f:
            model_columns = json.load(f)

        return model, scaler, encoder, model_columns

    except FileNotFoundError as e:
        st.error(f"Error: Could not find required file: {e.filename}. Please ensure all artifacts are in the root directory.")
        return None, None, None, None

model, scaler, encoder, model_columns = load_files()

# --- 3. UI LAYOUT ---
if model is not None:
    st.title("🏡 Airbnb Price Predictor")
    st.write("Enter your listing details below to estimate the nightly price.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Properties")
        accommodates = st.number_input("1. Accommodates (Guests)", min_value=1, value=2)
        bedrooms = st.number_input("2. Bedrooms", min_value=0, value=1)
        beds = st.number_input("3. Beds", min_value=1, value=1)
        
        # Updated: step=0.5 allows for 1.0, 1.5, 2.0, etc.
        bathrooms = st.number_input("4. Bathrooms", min_value=0.5, value=1.0, step=0.5)
        
        st.subheader("Location")
        
        # --- NEW INSTRUCTION TEXT ---
        st.info("ℹ️ **Note:** This model is trained only on the 30 specific cities listed below.")
        
        # 1. Strict City Selection
        selected_city_display = st.selectbox("5a. Select City (Required)", ALLOWED_CITIES_DISPLAY)
        
        # 2. Optional Street Address
        street_address = st.text_input("5b. Street Address (Optional)", 
                                      placeholder="e.g., 100 Main Street")
        
    with col2:
        st.subheader("Quality & Booking")
        review_scores_cleanliness = st.slider("6. Cleanliness Score (1-5)", 1.0, 5.0, 4.0, step=0.25)
        review_scores_location = st.slider("7. Location Score (1-5)", 1.0, 5.0, 4.0, step=0.25)
        
        instant_bookable = st.selectbox("8. Instant Bookable", ["Yes", "No"])
        instant_bookable_nb = 1 if instant_bookable == "Yes" else 0 
        
        property_categories = [
            'Entire Apartment/Condo', 'Entire House', 'Private Room', 
            'Hotel/Resort' 
        ]
        property_category = st.selectbox("9. Property Type Category", property_categories)

    st.markdown("---")
    
    # --- 4. PREDICTION LOGIC ---
    if st.button("Predict Price 💰", use_container_width=True):

        # --- NEW: Get the exact trained value from the display selection ---
        search_city_value = CITY_MAPPING[selected_city_display]
        
        # A. SMART GEOCODING
        if street_address:
            search_query = f"{street_address}, {search_city_value}"
        else:
            search_query = search_city_value
            
        st.info(f"📍 Locating: **{search_query}**")
            
        with st.spinner('Calculating coordinates...'):
            try:
                location = geolocator.geocode(search_query, timeout=10)
                
                if location is None:
                    st.error("Address not found. Please check spelling or try just the city.")
                    st.stop()
                
                # Check if city matches
                city_check = selected_city_display.split(",")[0].lower() 
                if city_check not in str(location).lower():
                     st.warning(f"⚠️ Note: We found a location at '{location}', which might not be in {selected_city_display}. Please verify the map below.")

                latitude = location.latitude
                longitude = location.longitude
                
                st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
                
            except Exception as e:
                st.error("Geocoding failed. Please try again.")
                st.stop()
        
        # B. PREPARE INPUT DATA
        raw_input_data = pd.DataFrame({
            "log_accommodates": [np.log(accommodates)],
            "log_beds_imputed": [np.log(beds)],
            "log_bedrooms_imputed": [np.log(bedrooms)],
            "log_bathrooms_imputed": [np.log(bathrooms)],
            "latitude": [latitude], 
            "longitude": [longitude], 
            "review_scores_cleanliness": [review_scores_cleanliness],
            "review_scores_location": [review_scores_location],
            "instant_bookable": [instant_bookable_nb],
            CATEGORICAL_FEATURE: [property_category]
        })

        try:
            # C. SCALING & ENCODING
            raw_input_data[SCALING_FEATURES] = scaler.transform(raw_input_data[SCALING_FEATURES])
            ohe_output = encoder.transform(raw_input_data[[CATEGORICAL_FEATURE]])
            ohe_col_names = encoder.get_feature_names_out([CATEGORICAL_FEATURE])
            ohe_df = pd.DataFrame(ohe_output, columns=ohe_col_names)
            
            features_to_keep = [c for c in raw_input_data.columns if c not in [CATEGORICAL_FEATURE]]
            final_df = pd.concat([raw_input_data[features_to_keep].reset_index(drop=True), ohe_df], axis=1)
            
            input_data = final_df.reindex(columns=model_columns, fill_value=0)

            # D. PREDICT
            log_prediction = model.predict(input_data)[0]
            price_prediction = np.exp(log_prediction)
            
            st.success(f"### Estimated Price: **${price_prediction:,.2f}** per night")
            
        except Exception as e:
            st.error("Prediction Logic Failed.")
            st.exception(e)