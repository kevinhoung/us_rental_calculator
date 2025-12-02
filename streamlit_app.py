import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from geopy.geocoders import Nominatim
import pydeck as pdk
from google.cloud import bigquery
from google.oauth2 import service_account

# --- 1. CONFIGURATION & CONSTANTS ---

# Initialize Geocoder
geolocator = Nominatim(user_agent="us_rental_calculator_app") 

# Define the features that were numerically scaled (StandardScaler/RobustScaler)
# This list MUST match the final numerical columns used in X_train_preprocessed.
# Note: It includes the log-transformed feature names!
SCALING_FEATURES = ["latitude", "longitude", "review_scores_cleanliness", "review_scores_location",
                    "instant_bookable", "log_beds_imputed", "log_accommodates",
                    "log_bedrooms_imputed", "log_bathrooms_imputed"]

# Define the CATEGORICAL feature that was One-Hot Encoded
CATEGORICAL_FEATURE = 'property_category'

# [CRITICAL] THE EXACT LIST OF 30 TRAINING CITIES
CITY_MAPPING = {
    # CLEAN DISPLAY NAME: EXACT TRAINED VALUE (Must match original DF)
    "Washington DC": "Washington Dc, DC",
    "New York City": "New York City, NY",
    "Los Angeles, CA": "Los Angeles, CA",
    "San Francisco, CA": "San Francisco, CA",
    "Las Vegas (Clark Co.)": "Clark County, NV",
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

# Color mapping for property categories (RGB values)
PROPERTY_COLORS = {
    'Entire Apartment/Condo': [65, 105, 225],      # Royal Blue
    'Entire House': [34, 139, 34],                  # Forest Green
    'Private Room': [255, 165, 0],                  # Orange
    'Hotel/Resort': [138, 43, 226],                 # Blue Violet
}

# --- BIGQUERY AUTHENTICATION ---
@st.cache_resource
def get_bigquery_client():
    """
    Initialize BigQuery client using service account credentials.
    Works with Streamlit Secrets for production, or local JSON file for development.
    """
    try:
        # Option 1: Try Streamlit Secrets (for production/Streamlit Cloud)
        if 'bigquery' in st.secrets and 'credentials' in st.secrets['bigquery']:
            credentials_info = json.loads(st.secrets['bigquery']['credentials'])
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            project_id = credentials_info.get('project_id', 'airbnb-dash-479208')
            client = bigquery.Client(credentials=credentials, project=project_id)
            return client
        
        # Option 2: Try local JSON file (for local development)
        try:
            credentials = service_account.Credentials.from_service_account_file(
                'service-account-key.json'
            )
            client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            return client
        except FileNotFoundError:
            pass
        
        # Option 3: Use default credentials (if gcloud auth is set up)
        client = bigquery.Client(project='airbnb-dash-479208')
        return client
        
    except Exception as e:
        st.warning(f"BigQuery authentication failed: {str(e)}")
        return None

# --- QUERY NEARBY LISTINGS FROM BIGQUERY ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def query_nearby_listings(latitude, longitude, radius_km=5, limit=20):
    """
    Query BigQuery for nearby Airbnb listings within radius
    Returns DataFrame with: latitude, longitude, price, property_category
    """
    client = get_bigquery_client()
    
    if client is None:
        return pd.DataFrame()  # Return empty if auth failed
    
    try:
        # SQL query to find nearby listings
        query = f"""
        SELECT 
            latitude,
            longitude,
            price,
            property_category,
            accommodates,
            bedrooms,
            bathrooms
        FROM 
            `airbnb-dash-479208.US_all.us_master_cleaned_final_v2`
        WHERE 
            ST_DISTANCE(
                ST_GEOGPOINT(longitude, latitude),
                ST_GEOGPOINT({longitude}, {latitude})
            ) <= {radius_km * 1000}  -- Convert km to meters
            AND price IS NOT NULL
            AND price > 0
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL
        LIMIT {limit}
        """
        
        # Execute query
        query_job = client.query(query)
        results = query_job.result()
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in results])
        
        if df.empty:
            return df
        
        # property_category is already in the correct format from BigQuery, no mapping needed
        return df
        
    except Exception as e:
        st.warning(f"Could not load competitor listings: {str(e)}")
        return pd.DataFrame()

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
            
            # --- MAP WITH COMPETITOR LISTINGS ---
            st.markdown("---")
            st.subheader("🗺️ Market Comparison Map")
            
            # Query nearby listings
            with st.spinner('Loading nearby competitor listings...'):
                competitor_df = query_nearby_listings(latitude, longitude, radius_km=5, limit=100)
            
            if not competitor_df.empty:
                # Prepare competitor data for map
                competitor_df['color'] = competitor_df['property_category'].map(
                    lambda x: PROPERTY_COLORS.get(x, [128, 128, 128])  # Default gray if category not found
                ).apply(lambda x: x + [200])  # Add alpha channel (semi-transparent)
                
                # Scale bubble size based on price (normalize to reasonable range)
                min_price = competitor_df['price'].min()
                max_price = competitor_df['price'].max()
                price_range = max_price - min_price if max_price > min_price else 1
                
                # Scale radius from 30 to 150 meters based on price
                competitor_df['radius'] = (
                    (competitor_df['price'] - min_price) / price_range * 120 + 30
                ).clip(30, 150)
                
                # Create prediction point data (golden star icon)
                prediction_data = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'price': [price_prediction],
                    'property_category': ['Your Prediction'],
                    'radius': [200],  # Larger radius for visibility
                    'color': [[255, 215, 0, 255]]  # Gold color, fully opaque
                })
                
                # Create competitor listings layer (colored bubbles)
                competitor_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=competitor_df,
                    id='competitor-listings',
                    get_position='[longitude, latitude]',
                    get_fill_color='color',
                    get_radius='radius',
                    pickable=True,  # Enable hover
                    auto_highlight=True,
                    radius_min_pixels=5,
                    radius_max_pixels=50,
                    radius_scale=1,
                )
                
                # Create prediction layer (golden star - using large circle with star emoji on top)
                # Base circle layer (large gold circle)
                prediction_circle_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=prediction_data,
                    id='prediction-circle',
                    get_position='[longitude, latitude]',
                    get_fill_color=[255, 215, 0, 255],  # Gold color
                    get_radius=250,  # Large radius in meters
                    pickable=True,
                    auto_highlight=True,
                    radius_min_pixels=20,
                    radius_max_pixels=100,
                    stroked=True,
                    get_line_color=[255, 140, 0, 255],  # Darker gold outline
                    line_width_min_pixels=4,
                )
                
                # Star emoji text layer on top
                prediction_text_layer = pdk.Layer(
                    'TextLayer',
                    data=prediction_data,
                    id='prediction-star',
                    get_position='[longitude, latitude]',
                    get_text='⭐',
                    get_color=[255, 255, 255, 255],  # White star
                    get_size=30,
                    get_angle=0,
                    get_text_anchor='middle',
                    get_alignment_baseline='center',
                    pickable=False,
                )
                
                # Tooltip for hover
                tooltip = {
                    "html": """
                    <b>{property_category}</b><br/>
                    Price: <b>${price:,.0f}</b> per night
                    """,
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white",
                        "padding": "8px",
                        "borderRadius": "5px",
                        "fontSize": "12px"
                    }
                }
                
                # Create map
                view_state = pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                )
                
                deck = pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=view_state,
                    layers=[competitor_layer, prediction_circle_layer, prediction_text_layer],  # Competitors, then prediction circle, then star on top
                    tooltip=tooltip,
                )
                
                st.pydeck_chart(deck, use_container_width=True)
                
                # Legend for property types
                st.markdown("---")
                st.subheader("📊 Legend")
                
                # Create a more visual legend
                legend_cols = st.columns(5)
                with legend_cols[0]:
                    st.markdown("**⭐ Your Prediction**")
                    st.markdown("🟡 Gold Star")
                with legend_cols[1]:
                    st.markdown("**🏢 Entire Apartment/Condo**")
                    st.markdown("🔵 Blue")
                with legend_cols[2]:
                    st.markdown("**🏠 Entire House**")
                    st.markdown("🟢 Green")
                with legend_cols[3]:
                    st.markdown("**🚪 Private Room**")
                    st.markdown("🟠 Orange")
                with legend_cols[4]:
                    st.markdown("**🏨 Hotel/Resort**")
                    st.markdown("🟣 Purple")
                
                st.caption("💡 **Tip:** Bubble size represents price - larger bubbles = higher prices. Hover over any point to see details. The ⭐ gold star shows your predicted price.")
                
                # Summary stats
                st.markdown("---")
                st.subheader("📈 Market Comparison")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nearby Listings", len(competitor_df))
                with col2:
                    avg_price = competitor_df['price'].mean()
                    st.metric("Avg Market Price", f"${avg_price:,.0f}")
                with col3:
                    st.metric("Your Prediction", f"${price_prediction:,.0f}")
                with col4:
                    diff = price_prediction - avg_price
                    pct_diff = (diff / avg_price * 100) if avg_price > 0 else 0
                    st.metric("vs Market", f"${diff:,.0f}", delta=f"{pct_diff:.1f}%")
                
            else:
                st.info("No competitor listings found nearby. Try a different location or check BigQuery connection.")
                # Still show prediction point on map with star
                prediction_data = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'price': [price_prediction],
                    'property_category': ['Your Prediction'],
                })
                
                prediction_circle_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=prediction_data,
                    get_position='[longitude, latitude]',
                    get_fill_color=[255, 215, 0, 255],  # Gold color
                    get_radius=250,
                    pickable=True,
                    radius_min_pixels=20,
                    radius_max_pixels=100,
                    stroked=True,
                    get_line_color=[255, 140, 0, 255],
                    line_width_min_pixels=4,
                )
                
                prediction_text_layer = pdk.Layer(
                    'TextLayer',
                    data=prediction_data,
                    get_position='[longitude, latitude]',
                    get_text='⭐',
                    get_color=[255, 255, 255, 255],
                    get_size=30,
                    get_angle=0,
                    get_text_anchor='middle',
                    get_alignment_baseline='center',
                )
                
                view_state = pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                )
                
                deck = pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=view_state,
                    layers=[prediction_circle_layer, prediction_text_layer],
                )
                
                st.pydeck_chart(deck, use_container_width=True)
            
        except Exception as e:
            st.error("Prediction Logic Failed.")
            st.exception(e)