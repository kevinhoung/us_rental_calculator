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
import base64
import os

# --- 1. CONFIGURATION & CONSTANTS ---

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Pricision AI",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize Geocoder
geolocator = Nominatim(user_agent="us_rental_calculator_app") 

# Define the features that were numerically scaled (StandardScaler/RobustScaler)
SCALING_FEATURES = ["latitude", "longitude", "review_scores_cleanliness", "review_scores_location",
                    "instant_bookable", "log_beds_imputed", "log_accommodates",
                    "log_bedrooms_imputed", "log_bathrooms_imputed"]

# Define the CATEGORICAL feature that was One-Hot Encoded
CATEGORICAL_FEATURE = 'property_category'

# [CRITICAL] THE EXACT LIST OF 30 TRAINING CITIES
CITY_MAPPING = {
    "Los Angeles, CA": "Los Angeles, CA",
    "Oakland, CA": "Oakland, CA",
    "Pacific Grove, CA": "Pacific Grove, CA",
    "San Diego, CA": "San Diego, CA",
    "San Francisco, CA": "San Francisco, CA",
    "San Mateo County, CA": "San Mateo County, CA",
    "Santa Clara County, CA": "Santa Clara County, CA",
    "Santa Cruz County, CA": "Santa Cruz County, CA",
    "Denver, CO": "Denver, CO",
    "Washington DC": "Washington Dc, DC",
    "Broward County, FL": "Broward County, FL",
    "Hawaii, HI": "Hawaii, HI",
    "New Orleans, LA": "New Orleans, LA",
    "Boston, MA": "Boston, MA",
    "Cambridge, MA": "Cambridge, MA",
    "Twin Cities, MN (MSA)": "Twin Cities MSA, MN",
    "Bozeman, MT": "Bozeman, MT",
    "Asheville, NC": "Asheville, NC",
    "Jersey City, NJ": "Jersey City, NJ",
    "Las Vegas (Clark Co.)": "Clark County, NV",
    "New York City": "New York City, NY",
    "Rochester, NY": "Rochester, NY",
    "Columbus, OH": "Columbus, OH",
    "Portland, OR": "Portland, OR",
    "Rhode Island, RI": "Rhode Island, RI",
    "Nashville, TN": "Nashville, TN",
    "Austin, TX": "Austin, TX",
    "Dallas, TX": "Dallas, TX",
    "Fort Worth, TX": "Fort Worth, TX",
    "Seattle, WA": "Seattle, WA",
}

ALLOWED_CITIES_DISPLAY = list(CITY_MAPPING.keys())

# Color mapping for property categories (RGB values)
PROPERTY_COLORS = {
    'Entire Apartment/Condo': [65, 105, 225],      # Royal Blue
    'Entire House': [34, 139, 34],                  # Forest Green
    'Private Room': [255, 165, 0],                  # Orange
    'Hotel/Resort': [138, 43, 226],                 # Blue Violet
}

# Emoji mapping for property categories
PROPERTY_EMOJIS = {
    'Entire Apartment/Condo': '🏢',
    'Entire House': '🏠',
    'Private Room': '🚪',
    'Hotel/Resort': '🏨',
}

# --- BIGQUERY AUTHENTICATION (SIMPLIFIED) ---
@st.cache_resource
def get_bigquery_client():
    """
    Initialize BigQuery client using Streamlit Secrets.
    """
    try:
        # Option 1: Streamlit Cloud (Secrets)
        if 'bigquery' in st.secrets:
            # We use the flattened dictionary from secrets.toml directly
            creds_dict = dict(st.secrets['bigquery'])
            
            # Create credentials from the dictionary
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            
            # Create and return client
            client = bigquery.Client(credentials=credentials, project=creds_dict['project_id'])
            return client
            
        # Option 2: Local Development (fallback to local json file if secrets fail)
        # This is useful if you are running locally and haven't set up secrets.toml yet
        elif os.path.exists('service-account-key.json'):
            credentials = service_account.Credentials.from_service_account_file('service-account-key.json')
            client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            return client
            
        else:
            st.error("❌ BigQuery authentication failed. 'bigquery' section not found in secrets.toml.")
            return None

    except Exception as e:
        st.error(f"❌ BigQuery Connection Error: {str(e)}")
        return None

# --- QUERY NEARBY LISTINGS FROM BIGQUERY ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def query_nearby_listings(latitude, longitude, radius_km=5, limit=20):
    """
    Query BigQuery for nearby Airbnb listings within radius
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
    # --- HEADER SECTION (UPDATED FOR BRANDING) ---
    try:
        # Display the large logo banner
        st.image("images/logo.png", use_container_width=True)
    except Exception:
        st.title("Pricision AI") # Fallback

    st.write("") # Spacing

    # Centered Description
    st.markdown("""
    <div style="text-align: center; font-size: 1.1em; color: #555; margin-bottom: 20px;">
    <b>Pricision AI</b> is a proprietary machine learning engine that delivers the optimal nightly 
    rate for your short-term rental with unmatched accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    # How-To Guide in Expander
    with st.expander("📖 How-To Guide: Get Your Optimal Rate"):
        st.markdown("""
        The power of Pricision is in its simplicity:
        
        1. **Input Listing Details:** Enter location, bedrooms, and bathrooms.
        2. **Select Preferences:** Choose property type and booking settings.
        3. **Click 'Get Optimal Rate':** Instantly receive your AI-optimized prediction and competitor map.
        
        💡 **Pro Tip:** The more accurate your inputs, the more precise your price prediction will be!
        """)
    
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Properties")
        accommodates = st.number_input("1. Accommodates (Guests)", min_value=1, value=2)
        bedrooms = st.number_input("2. Bedrooms", min_value=0, value=1)
        beds = st.number_input("3. Beds", min_value=1, value=1)
        bathrooms = st.number_input("4. Bathrooms", min_value=0.5, value=1.0, step=0.5)
        
        st.subheader("Location")
        
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
        
        # --- NEURAL PRICING BUTTON ---
        st.write("") # Spacing
        st.subheader("Neural Pricing")
        
        # Center the button using columns
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        
        with button_col2:
            # Use streamlit-image-button component for image button
            try:
                from streamlit_image_button import st_image_button
                
                # Image path - ensure brain2.png is in images/ folder
                image_path = "images/brain2.png"
                
                # Image button - returns True when clicked
                predict_button = st_image_button(
                    "Get Optimal Rate",
                    image_path,
                    use_container_width=True
                )
            except ImportError:
                # Fallback: Show image above regular button if package not available
                st.image("images/brain2.png", width=200, use_container_width=True)
                predict_button = st.button(
                    "Get Optimal Rate", 
                    use_container_width=True,
                    type="primary"
                )
            except Exception:
                # Fallback if image file is missing entirely
                predict_button = st.button("Get Optimal Rate", type="primary", use_container_width=True)
        
            # Loading animation placeholder (right under the button, centered)
            loading_placeholder = st.empty()
            
            # Add some spacing to move animation up
            st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- 4. PREDICTION LOGIC ---
    # Check if button was clicked
    if predict_button:
        
        # Show AI thinking animation right under the button (centered and moved up)
        with loading_placeholder.container():
            st.markdown("""
            <div style="text-align: center; padding: 5px; margin-top: -15px;">
                <div style="font-size: 32px; animation: pulse 1.5s ease-in-out infinite; margin-bottom: 5px;">🤖</div>
                <p style="margin: 2px 0; font-size: 13px; font-weight: 500;">AI is analyzing...</p>
                <p style="margin: 2px 0; font-size: 11px; color: #666;">Calculating optimal price</p>
            </div>
            <style>
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.1); opacity: 0.8; }
            }
            </style>
            """, unsafe_allow_html=True)
        
        # --- NEW: Get the exact trained value from the display selection ---
        search_city_value = CITY_MAPPING[selected_city_display]
        
        # A. SMART GEOCODING
        if street_address:
            search_query = f"{street_address}, {search_city_value}"
        else:
            search_query = search_city_value
            
        with st.spinner('🤖 AI is calculating coordinates...'):
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
            with st.spinner('🤖 AI is processing your listing details...'):
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
            
            # Clear loading animation
            loading_placeholder.empty()
            
            # Display location and price on one line
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.info(f"📍 Locating: **{search_query}**")
            with info_col2:
                st.success(f"֎ Estimated Price: **${price_prediction:,.2f}** per night")
            
            # --- MAP WITH COMPETITOR LISTINGS ---
            st.markdown("---")
            
            # Query nearby listings
            with st.spinner('Loading nearby competitor listings...'):
                competitor_df = query_nearby_listings(latitude, longitude, radius_km=5, limit=200)
            
            # Debug: Show what we got
            if competitor_df.empty:
                client = get_bigquery_client()
                if client is None:
                    st.warning("⚠️ BigQuery authentication failed. Check your credentials.")
                else:
                    st.info(f"ℹ️ No competitor listings found within 5km of this location.")
            
            if not competitor_df.empty:
                # Prepare competitor data for map
                competitor_df['emoji'] = competitor_df['property_category'].map(
                    lambda x: PROPERTY_EMOJIS.get(x, '📍')
                )
                
                competitor_df['price_formatted'] = competitor_df['price'].apply(lambda x: f"${x:,.0f}")
                
                # Add color based on property category
                competitor_df['color'] = competitor_df['property_category'].map(
                    lambda x: PROPERTY_COLORS.get(x, [128, 128, 128])
                ).apply(lambda x: x + [200])  # Add alpha
                
                # Scale radius based on price
                min_price = competitor_df['price'].min()
                max_price = competitor_df['price'].max()
                price_range = max_price - min_price if max_price > min_price else 1
                competitor_df['radius'] = (
                    (competitor_df['price'] - min_price) / price_range * 120 + 30
                ).clip(30, 150)
                
                # Create prediction point data (golden star)
                prediction_data = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'price': [price_prediction],
                    'price_formatted': [f"${price_prediction:,.0f}"],
                    'property_category': ['Your Prediction'],
                    'emoji': ['⭐'], 
                    'radius': [200],
                    'color': [[255, 215, 0, 255]] 
                })
                
                # Competitor Layer
                competitor_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=competitor_df,
                    id='competitor-listings',
                    get_position='[longitude, latitude]',
                    get_fill_color='color',
                    get_radius='radius',
                    pickable=True,
                    auto_highlight=True,
                    radius_min_pixels=5,
                    radius_max_pixels=50,
                    radius_scale=1,
                )
                
                # Prediction Circle Layer
                prediction_circle_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=prediction_data,
                    id='prediction-circle',
                    get_position='[longitude, latitude]',
                    get_fill_color=[255, 215, 0, 255], 
                    get_radius=250, 
                    pickable=True,
                    auto_highlight=True,
                    radius_min_pixels=20,
                    radius_max_pixels=100,
                    stroked=True,
                    get_line_color=[255, 140, 0, 255],
                    line_width_min_pixels=4,
                )
                
                # Star Text Layer
                prediction_text_layer = pdk.Layer(
                    'TextLayer',
                    data=prediction_data,
                    id='prediction-star',
                    get_position='[longitude, latitude]',
                    get_text='emoji', 
                    get_color=[255, 255, 255, 255], 
                    get_size=40, 
                    get_angle=0,
                    get_text_anchor='middle',
                    get_alignment_baseline='center',
                    pickable=False,
                    size_scale=1,
                    size_min_pixels=30,
                    size_max_pixels=50,
                )
                
                # Tooltip
                tooltip = {
                    "html": """
                    <b>{emoji} {property_category}</b><br/>
                    Price: <b>{price_formatted}</b> per night
                    """,
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white",
                        "padding": "8px",
                        "borderRadius": "5px",
                        "fontSize": "12px"
                    }
                }
                
                # View State
                view_state = pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                )
                
                # Get Mapbox token
                mapbox_token = None
                try:
                    if 'mapbox' in st.secrets and 'token' in st.secrets['mapbox']:
                        mapbox_token = st.secrets['mapbox']['token']
                        pdk.settings.mapbox_key = mapbox_token
                except Exception:
                    pass
                
                if mapbox_token:
                    map_style = 'mapbox://styles/mapbox/streets-v12'
                else:
                    map_style = 'road'
                
                deck = pdk.Deck(
                    map_style=map_style,
                    initial_view_state=view_state,
                    layers=[competitor_layer, prediction_circle_layer, prediction_text_layer],
                    tooltip=tooltip,
                )
                
                # Metrics
                st.subheader("📈 Market Comparison")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nearby Listings", len(competitor_df))
                with col2:
                    avg_price = competitor_df['price'].mean()
                    st.metric("Avg Market Price", f"${avg_price:,.0f}")
                with col3:
                    st.metric("Prediction", f"${price_prediction:,.0f}")
                with col4:
                    diff = price_prediction - avg_price
                    pct_diff = (diff / avg_price * 100) if avg_price > 0 else 0
                    st.metric("Versus Market", f"${diff:,.0f}", delta=f"{pct_diff:.1f}%")
                
                # Legend and Map
                legend_col, map_col = st.columns([1, 3])
                
                with legend_col:
                    st.markdown("### 📊 Legend")
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">⭐</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #FFD700; border: 2px solid #FF8C00; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🏢</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #4169E1; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🏠</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #228B22; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🚪</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #FFA500; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🏨</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #8A2BE2; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    st.caption("💡 **Tip:** Circle size represents price.")
                
                with map_col:
                    st.subheader("🗺️ Market Comparison Map")
                    st.pydeck_chart(deck, use_container_width=True)
                
            else:
                # No data fallback map
                st.info("No competitor listings found nearby. Showing your location only.")
                
                prediction_data = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'price': [price_prediction],
                    'property_category': ['Your Prediction'],
                    'emoji': ['⭐'],
                })
                
                prediction_circle_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=prediction_data,
                    get_position='[longitude, latitude]',
                    get_fill_color=[255, 215, 0, 255],
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
                    get_text='emoji', 
                    get_color=[255, 255, 255, 255],
                    get_size=40, 
                    pickable=False,
                )
                
                view_state = pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                )
                
                deck = pdk.Deck(
                    map_style='road',
                    initial_view_state=view_state,
                    layers=[prediction_circle_layer, prediction_text_layer],
                )
                
                st.pydeck_chart(deck, use_container_width=True)
            
        except Exception as e:
            st.error("Prediction Logic Failed.")
            st.exception(e)