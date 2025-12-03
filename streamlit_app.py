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
# Organized by state, then alphabetically within each state
CITY_MAPPING = {
    # CLEAN DISPLAY NAME: EXACT TRAINED VALUE (Must match original DF)
    # California (CA)
    "Los Angeles, CA": "Los Angeles, CA",
    "Oakland, CA": "Oakland, CA",
    "Pacific Grove, CA": "Pacific Grove, CA",
    "San Diego, CA": "San Diego, CA",
    "San Francisco, CA": "San Francisco, CA",
    "San Mateo County, CA": "San Mateo County, CA",
    "Santa Clara County, CA": "Santa Clara County, CA",
    "Santa Cruz County, CA": "Santa Cruz County, CA",
    
    # Colorado (CO)
    "Denver, CO": "Denver, CO",
    
    # District of Columbia (DC)
    "Washington DC": "Washington Dc, DC",
    
    # Florida (FL)
    "Broward County, FL": "Broward County, FL",
    
    # Hawaii (HI)
    "Hawaii, HI": "Hawaii, HI",
    
    # Louisiana (LA)
    "New Orleans, LA": "New Orleans, LA",
    
    # Massachusetts (MA)
    "Boston, MA": "Boston, MA",
    "Cambridge, MA": "Cambridge, MA",
    
    # Minnesota (MN)
    "Twin Cities, MN (MSA)": "Twin Cities MSA, MN",
    
    # Montana (MT)
    "Bozeman, MT": "Bozeman, MT",
    
    # North Carolina (NC)
    "Asheville, NC": "Asheville, NC",
    
    # New Jersey (NJ)
    "Jersey City, NJ": "Jersey City, NJ",
    
    # Nevada (NV)
    "Las Vegas (Clark Co.)": "Clark County, NV",
    
    # New York (NY)
    "New York City": "New York City, NY",
    "Rochester, NY": "Rochester, NY",
    
    # Ohio (OH)
    "Columbus, OH": "Columbus, OH",
    
    # Oregon (OR)
    "Portland, OR": "Portland, OR",
    
    # Rhode Island (RI)
    "Rhode Island, RI": "Rhode Island, RI",
    
    # Tennessee (TN)
    "Nashville, TN": "Nashville, TN",
    
    # Texas (TX)
    "Austin, TX": "Austin, TX",
    "Dallas, TX": "Dallas, TX",
    "Fort Worth, TX": "Fort Worth, TX",
    
    # Washington (WA)
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

# --- HELPER FUNCTION: Create emoji icon atlas and mapping for IconLayer ---
@st.cache_data
def create_emoji_icon_atlas(emojis):
    """
    Create an icon atlas and mapping for pydeck IconLayer from emojis.
    Returns: (icon_atlas_url, icon_mapping)
    """
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    icon_size = 64
    num_icons = len(emojis)
    cols = int(np.ceil(np.sqrt(num_icons)))
    rows = int(np.ceil(num_icons / cols))
    
    # Create atlas image
    atlas_width = cols * icon_size
    atlas_height = rows * icon_size
    atlas_img = Image.new('RGBA', (atlas_width, atlas_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(atlas_img)
    
    # Try to load emoji font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", size=48)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", size=48)
        except:
            font = ImageFont.load_default()
    
    # Create icon mapping - use numeric keys (0, 1, 2, ...) instead of emoji strings
    icon_mapping = {}
    emoji_to_index = {}  # Map emoji to index
    
    for idx, emoji in enumerate(emojis):
        row = idx // cols
        col = idx % cols
        x = col * icon_size + icon_size // 2
        y = row * icon_size + icon_size // 2
        
        # Draw emoji
        bbox = draw.textbbox((0, 0), emoji, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x - text_width / 2 - bbox[0]
        text_y = y - text_height / 2 - bbox[1]
        draw.text((text_x, text_y), emoji, font=font, fill=(0, 0, 0, 255))
        
        # Create mapping for this icon using string key (pydeck requires string keys)
        icon_key = str(idx)
        icon_mapping[icon_key] = {
            'x': col * icon_size,
            'y': row * icon_size,
            'width': icon_size,
            'height': icon_size,
            'mask': True
        }
        emoji_to_index[emoji] = icon_key
    
    # Convert to base64 data URL
    buffered = io.BytesIO()
    atlas_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    icon_atlas_url = f"data:image/png;base64,{img_str}"
    
    return icon_atlas_url, icon_mapping, emoji_to_index

# --- BIGQUERY AUTHENTICATION ---
@st.cache_resource
def get_bigquery_client():
    """
    Initialize BigQuery client using service account credentials.
    Follows the official Streamlit pattern: https://docs.streamlit.io/develop/tutorials/databases/bigquery
    
    Works with:
    1. Local JSON file (for local development)
    2. Streamlit Secrets with 'gcp_service_account' key (recommended for Streamlit Cloud)
    3. Streamlit Secrets with 'bigquery.credentials' key (backward compatibility)
    """
    import os
    
    try:
        # Option 1: Try Streamlit Secrets - Official pattern using 'gcp_service_account'
        # This is the recommended approach per Streamlit docs
        if 'gcp_service_account' in st.secrets:
            try:
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"]
                )
                client = bigquery.Client(credentials=credentials)
                return client
            except Exception as e:
                st.warning(f"Failed to load credentials from 'gcp_service_account' secret: {str(e)}")
        
    except Exception as e:
        st.error(f"BigQuery authentication failed: {str(e)}")
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
    # Centered logo at the top - very large
    logo_col1, logo_col2, logo_col3 = st.columns([1, 3, 1])
    
    with logo_col2:
        st.image("images/logo.png", width=1200)  # Very large centered logo
    
    st.markdown("---")
    
    # Descriptive text above How-To Guide
    st.write("Pricision AI is a proprietary machine learning engine that delivers the optimal nightly " \
    "rate for your short-term rental with unmatched accuracy.")
    
    # How-To Guide
    st.markdown("### 📖 How-To Guide: Get Your Optimal Rate")
    st.markdown("""
    The power of Pricision is in its simplicity. You can get your rate estimate in just a few quick steps:
    
    1. **Input Listing Details:** Enter the basic information about your property (location, bedrooms, bathrooms).
    
    2. **Select Your Preferences:** Choose your property type, review scores, and booking preferences.
    
    3. **Click 'Predict Price':** Instantly receive your **AI-optimized nightly rate**
                long with a market comparison map showing nearby competitor listings.")
    
    💡 **Pro Tip:** The more accurate your inputs, the more precise your price prediction will be!
    """)
    
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
        
        # Predict Price section directly below Quality & Booking
        st.subheader("Neural Pricing")
        
        # Center the button using columns
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        
        with button_col2:
            # Use streamlit-image-button component for image button
            try:
                from streamlit_image_button import st_image_button
                
                # Image path - make sure brain.png is in the images folder
                image_path = "images/brain2.png"
                
                # Image button - returns True when clicked
                predict_button = st_image_button(
                    "Get Optimal Rate",
                    image_path,
                    use_container_width=True
                )
            except ImportError:
                # Fallback: Show image above regular button if package not available
                st.image("images/brain2.png", width=200, use_container_width=False)
                predict_button = st.button(
                    "Predict\nPrice", 
                    use_container_width=True,
                    type="primary"
                )
        
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
            latitude = None
            longitude = None
            
            # Try geocoding with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    location = geolocator.geocode(search_query, timeout=15)
                    
                    if location is None:
                        # If address not found, try just the city
                        if street_address:
                            st.warning(f"⚠️ Address '{street_address}' not found. Using city center coordinates.")
                            location = geolocator.geocode(search_city_value, timeout=15)
                            if location is None:
                                raise Exception(f"Could not geocode city: {search_city_value}")
                        else:
                            raise Exception(f"Could not geocode: {search_query}")
                    
                    # Check if city matches
                    city_check = selected_city_display.split(",")[0].lower() 
                    if city_check not in str(location).lower():
                         st.warning(f"⚠️ Note: We found a location at '{location}', which might not be in {selected_city_display}. Please verify the map below.")

                    latitude = location.latitude
                    longitude = location.longitude
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Wait before retry (exponential backoff)
                        import time
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        # All retries failed - use fallback coordinates
                        st.warning(f"⚠️ Geocoding service unavailable. Using approximate city center coordinates.")
                        # Fallback: Use approximate coordinates for major cities
                        # This is a simple fallback - you might want to add more cities
                        fallback_coords = {
                            "Los Angeles, CA": (34.0522, -118.2437),
                            "San Francisco, CA": (37.7749, -122.4194),
                            "New York City, NY": (40.7128, -74.0060),
                            "Chicago, IL": (41.8781, -87.6298),
                            "Boston, MA": (42.3601, -71.0589),
                            "Seattle, WA": (47.6062, -122.3321),
                            "Austin, TX": (30.2672, -97.7431),
                            "Denver, CO": (39.7392, -104.9903),
                            "Portland, OR": (45.5152, -122.6784),
                            "Nashville, TN": (36.1627, -86.7816),
                        }
                        
                        if search_city_value in fallback_coords:
                            latitude, longitude = fallback_coords[search_city_value]
                            st.info(f"📍 Using approximate coordinates for {selected_city_display}")
                        else:
                            # Generic fallback - use a default US center
                            st.error(f"❌ Geocoding failed after {max_retries} attempts: {str(e)}")
                            st.info("💡 Please try again in a moment, or use just the city name without a street address.")
                            st.stop()
            
            if latitude is None or longitude is None:
                st.error("❌ Could not determine coordinates. Please try again.")
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
                # Check if it's an auth issue or no data issue
                client = get_bigquery_client()
                if client is None:
                    st.warning("⚠️ BigQuery authentication failed. Check your credentials.")
                else:
                    st.info(f"ℹ️ No competitor listings found within 5km of this location. Found {len(competitor_df)} listings.")
            
            if not competitor_df.empty:
                # Prepare competitor data for map
                # Add emoji for each property category
                competitor_df['emoji'] = competitor_df['property_category'].map(
                    lambda x: PROPERTY_EMOJIS.get(x, '📍')  # Default pin emoji if category not found
                )
                
                # Format price for tooltip display
                competitor_df['price_formatted'] = competitor_df['price'].apply(lambda x: f"${x:,.0f}")
                
                # Scale emoji size based on price (normalize to reasonable range)
                min_price = competitor_df['price'].min()
                max_price = competitor_df['price'].max()
                price_range = max_price - min_price if max_price > min_price else 1
                
                # Scale emoji size from 20 to 50 pixels based on price (larger for visibility)
                competitor_df['emoji_size'] = (
                    (competitor_df['price'] - min_price) / price_range * 30 + 20
                ).clip(20, 50).astype(int)
                
                # Create prediction point data (golden star icon)
                prediction_data = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'price': [price_prediction],
                    'price_formatted': [f"${price_prediction:,.0f}"],
                    'property_category': ['Your Prediction'],
                    'emoji': ['⭐'],  # Add star emoji as a column
                    'radius': [200],  # Larger radius for visibility
                    'color': [[255, 215, 0, 255]]  # Gold color, fully opaque
                })
                
                # Create competitor listings layer (using ScatterplotLayer with emoji in tooltip)
                # Since IconLayer with data URLs is causing issues, we'll use ScatterplotLayer
                # with colored circles and show emoji in tooltip
                competitor_df['emoji'] = competitor_df['emoji'].astype(str)
                competitor_df = competitor_df.reset_index(drop=True)
                
                # Add color based on property category
                competitor_df['color'] = competitor_df['property_category'].map(
                    lambda x: PROPERTY_COLORS.get(x, [128, 128, 128])
                ).apply(lambda x: x + [200])  # Add alpha channel
                
                # Scale radius based on price
                min_price = competitor_df['price'].min()
                max_price = competitor_df['price'].max()
                price_range = max_price - min_price if max_price > min_price else 1
                competitor_df['radius'] = (
                    (competitor_df['price'] - min_price) / price_range * 120 + 30
                ).clip(30, 150)
                
                # Create ScatterplotLayer for competitor listings
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
                
                # Note: TextLayer doesn't reliably render emojis in pydeck
                # So we'll use ScatterplotLayer with colored circles
                # and show emojis in the tooltip instead
                # The colored circles will represent property types, size = price
                
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
                    get_text='emoji',  # Use column name instead of hardcoded string
                    get_color=[255, 255, 255, 255],  # White star
                    get_size=40,  # Larger size for visibility
                    get_angle=0,
                    get_text_anchor='middle',
                    get_alignment_baseline='center',
                    pickable=False,
                    size_scale=1,
                    size_min_pixels=30,
                    size_max_pixels=50,
                )
                
                # Tooltip for hover - include emoji in tooltip
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
                
                # Create map
                view_state = pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                )
                
                # Get Mapbox token from secrets (if available)
                mapbox_token = None
                try:
                    if 'mapbox' in st.secrets and 'token' in st.secrets['mapbox']:
                        mapbox_token = st.secrets['mapbox']['token']
                        # Set the Mapbox API key for pydeck globally (must be set before creating deck)
                        pdk.settings.mapbox_key = mapbox_token
                except Exception as e:
                    pass
                
                # Use Mapbox style if token is available, otherwise use default
                if mapbox_token:
                    # Try different Mapbox style - use streets which is more reliable
                    map_style = 'mapbox://styles/mapbox/streets-v12'
                    # Also set as environment variable (some pydeck versions need this)
                    import os
                    os.environ['MAPBOX_API_KEY'] = mapbox_token
                else:
                    map_style = 'road'  # Fallback to free style
                
                # Create deck with Mapbox configuration
                # Note: pdk.settings.mapbox_key must be set before creating the deck
                deck = pdk.Deck(
                    map_style=map_style,
                    initial_view_state=view_state,
                    layers=[competitor_layer, prediction_circle_layer, prediction_text_layer],  # Competitors (colored circles), then prediction circle, then star on top
                    tooltip=tooltip,
                )
                
                # Market Comparison metrics (above map)
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
                
                # Create column layout: legend on left, map on right
                legend_col, map_col = st.columns([1, 3])
                
                with legend_col:
                    st.markdown("### 📊 Legend")
                    # Prediction: star = gold circle (all on one line)
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">⭐</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #FFD700; border: 2px solid #FF8C00; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    # Apartment/Condo: emoji = blue circle
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🏢</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #4169E1; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    # House: emoji = green circle
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🏠</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #228B22; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    # Private Room: emoji = orange circle
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🚪</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #FFA500; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    # Hotel/Resort: emoji = purple circle
                    st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span style="font-size: 20px; margin-right: 8px;">🏨</span><span style="margin-right: 8px;">=</span><div style="width: 20px; height: 20px; background-color: #8A2BE2; border-radius: 50%;"></div></div>', unsafe_allow_html=True)
                    st.caption("💡 **Tip:** Circle size represents price - larger circles = higher prices. Hover over any point to see details.")
                
                with map_col:
                    st.subheader("🗺️ Market Comparison Map")
                    st.pydeck_chart(deck, use_container_width=True)
                
            else:
                st.info("No competitor listings found nearby. Try a different location or check BigQuery connection.")
                # Still show prediction point on map with star
                prediction_data = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'price': [price_prediction],
                    'property_category': ['Your Prediction'],
                    'emoji': ['⭐'],  # Add star emoji as a column
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
                    get_text='emoji',  # Use column name instead of hardcoded string
                    get_color=[255, 255, 255, 255],
                    get_size=40,  # Larger size for visibility
                    get_angle=0,
                    get_text_anchor='middle',
                    get_alignment_baseline='center',
                    size_scale=1,
                    size_min_pixels=30,
                    size_max_pixels=50,
                )
                
                view_state = pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=12,
                    pitch=50,
                )
                
                # Get Mapbox token from secrets (if available)
                mapbox_token = None
                try:
                    if 'mapbox' in st.secrets and 'token' in st.secrets['mapbox']:
                        mapbox_token = st.secrets['mapbox']['token']
                        # Set the Mapbox API key for pydeck globally
                        pdk.settings.mapbox_key = mapbox_token
                except Exception as e:
                    pass
                
                # Use Mapbox style if token is available, otherwise use default
                if mapbox_token:
                    # Try different Mapbox style - use streets which is more reliable
                    map_style = 'mapbox://styles/mapbox/streets-v12'
                    # Also set as environment variable (some pydeck versions need this)
                    import os
                    os.environ['MAPBOX_API_KEY'] = mapbox_token
                else:
                    map_style = 'road'  # Fallback to free style
                
                # Create deck with Mapbox configuration
                # Note: pdk.settings.mapbox_key must be set before creating the deck
                deck = pdk.Deck(
                    map_style=map_style,
                    initial_view_state=view_state,
                    layers=[prediction_circle_layer, prediction_text_layer],
                )
                
                st.pydeck_chart(deck, use_container_width=True)
            
        except Exception as e:
            st.error("Prediction Logic Failed.")
            st.exception(e)