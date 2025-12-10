import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import requests
import pydeck as pdk
from google.cloud import bigquery
from google.oauth2 import service_account
import base64
import os
import altair as alt
from sidebar import show_sidebar
from st_image_button import st_image_button

# Note: HomeHarvest requires Python 3.10+ due to typing syntax (list[dict] | None)
# On Python 3.9, it will fail silently and fall back to RentCast

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Pricision AI",
    page_icon="images/logo.png"  # Custom favicon
)

# Show sidebar (call this early, right after page config)
show_sidebar()

# --- HELPER: Load secrets from local TOML or Streamlit secrets ---
def get_secret(key_path, default=None):
    """
    Get secret value from either local .streamlit/secrets.toml (local dev) 
    or Streamlit secrets (production).
    
    Args:
        key_path: Dot-separated path (e.g., 'api_ninjas.api_key' or 'mapbox.token')
        default: Default value if not found
    
    Returns:
        Secret value or default
    """
    # Streamlit automatically loads .streamlit/secrets.toml when running locally
    # and uses cloud secrets in production. So st.secrets works for both cases.
    # However, we can explicitly check for local file if needed.
    
    # Try to load from local file first (for explicit local development)
    secrets_path = '.streamlit/secrets.toml'
    if os.path.exists(secrets_path):
        try:
            # Try Python 3.11+ tomllib first
            try:
                import tomllib  # type: ignore
                with open(secrets_path, 'rb') as f:
                    local_secrets = tomllib.load(f)
            except (ImportError, AttributeError):
                # Fallback to tomli for Python < 3.11
                try:
                    import tomli  # type: ignore
                    with open(secrets_path, 'rb') as f:
                        local_secrets = tomli.load(f)
                except ImportError:
                    # If neither available, use st.secrets (which auto-loads TOML)
                    local_secrets = None
            
            if local_secrets:
                # Navigate the nested dict using dot notation
                keys = key_path.split('.')
                value = local_secrets
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        # Not found in local, try st.secrets
                        break
                else:
                    # Successfully found in local secrets
                    return value
        except Exception:
            pass
    
    # Fallback to Streamlit secrets (auto-loads .streamlit/secrets.toml locally, or uses cloud secrets in production)
    try:
        keys = key_path.split('.')
        value = st.secrets
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    except Exception as e:
        # Debug: Only show error in development if needed
        # st.caption(f"Debug: Error loading secret '{key_path}': {str(e)}")
        return default

# --- 1. CONFIGURATION & CONSTANTS ---

# Mapbox Geocoding function (v6 API)
# Note: No caching - we want fresh results for each unique address
def geocode_with_mapbox(query, mapbox_token):
    """
    Geocode an address using Mapbox Geocoding API v6.
    Documentation: https://docs.mapbox.com/api/search/geocoding/
    
    Returns (latitude, longitude, city, state) or (None, None, None, None) if location data not available.
    """
    if not mapbox_token:
        return None, None, None, None
    
    if not query or not query.strip():
        return None, None, None, None
    
    try:
        # Mapbox Geocoding API v6 endpoint
        # Forward geocoding: https://api.mapbox.com/search/geocode/v6/forward
        url = "https://api.mapbox.com/search/geocode/v6/forward"
        params = {
            'q': query.strip(),  # Search text (required) - strip whitespace
            'access_token': mapbox_token,  # Access token (required)
            'limit': 1,  # Return only the best result
            'autocomplete': 'true',  # Enable autocomplete (default, but explicit)
            'country': 'US'  # Limit to US addresses for better accuracy
        }
        
        # If query contains a state abbreviation, add proximity bias
        # Extract state from query if possible (e.g., "123 Main St, Los Angeles, CA")
        if ', ' in query:
            parts = query.split(', ')
            if len(parts) >= 2:
                last_part = parts[-1].strip()
                # Check if last part is a state (2 letters)
                if len(last_part) == 2 and last_part.isalpha():
                    # Try to get approximate coordinates for the state to bias results
                    # This helps Mapbox prioritize results in that state
                    pass  # We'll add proximity if we have city coordinates
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Debug: Print the response structure (only in development)
        # st.write(f"Debug - Mapbox response: {json.dumps(data, indent=2)[:500]}")
        
        # v6 API response structure: features array with geometry.coordinates
        if data.get('features') and len(data['features']) > 0:
            feature = data['features'][0]
            geometry = feature.get('geometry', {})
            coordinates = geometry.get('coordinates')
            
            if not coordinates or len(coordinates) < 2:
                st.warning(f"⚠️ Invalid coordinates in Mapbox response for: {query}")
                return None, None, None, None
            
            # Mapbox returns [longitude, latitude], we need [latitude, longitude]
            longitude = float(coordinates[0])
            latitude = float(coordinates[1])
            
            # Validate coordinates are reasonable
            if not (-180 <= longitude <= 180) or not (-90 <= latitude <= 90):
                st.warning(f"⚠️ Coordinates out of range: lat={latitude}, lon={longitude} for query: {query}")
                return None, None, None, None
            
            # Debug output - show what was geocoded
            # Get the full address from the feature if available
            feature_name = feature.get('properties', {}).get('name', query)
            
            # Extract city and state from properties/context if available
            city = None
            state = None
            
            # Try different ways to extract city and state from Mapbox response
            properties = feature.get('properties', {})
            
            # Method 1: Try context array (v6 API structure)
            if 'context' in properties:
                context = properties.get('context', [])
                if isinstance(context, list):
                    for item in context:
                        if isinstance(item, dict):
                            # Look for place (city) and region (state)
                            item_type = item.get('type', '')
                            if 'place' in item_type.lower() and 'name' in item:
                                city = item['name']
                            elif 'region' in item_type.lower() and 'name' in item:
                                region_name = item['name']
                                # Check if it's a 2-letter state code
                                if len(region_name) == 2:
                                    state = region_name.upper()
                                # Could also be full state name, but we'll use abbreviation if available
            
            return latitude, longitude, city, state
        
        return None, None, None, None
    except requests.exceptions.HTTPError as e:
        # Handle specific HTTP errors
        if e.response.status_code == 401:
            st.error("❌ Mapbox authentication failed. Check your access token.")
        elif e.response.status_code == 429:
            st.warning("⚠️ Mapbox rate limit exceeded. Using fallback coordinates.")
        else:
            st.warning(f"⚠️ Mapbox geocoding error ({e.response.status_code}): {e.response.text[:100]}")
        return None, None, None, None
    except Exception as e:
        st.warning(f"⚠️ Geocoding error: {str(e)}")
        return None, None, None, None 

# API Ninjas Mortgage Rate API function
@st.cache_data(ttl=3600)  # Cache for 1 hour (rates don't change that frequently)
def get_apininjas_mortgage_rate(loan_term, api_key):
    """
    Fetch current mortgage rate from API Ninjas.
    Documentation: https://api-ninjas.com/api/mortgagerates
    
    Args:
        loan_term: Loan term in years (10, 15, 20, 30)
        api_key: API Ninjas API key
    
    Returns:
        float: Interest rate as decimal (e.g., 0.065 for 6.5%), or None if failed
    """
    if not api_key:
        return None
    
    try:
        # API Ninjas endpoint
        url = "https://api.api-ninjas.com/v1/mortgagerates"
        
        # API Ninjas uses X-Api-Key header for authentication
        headers = {
            'X-Api-Key': api_key
        }
        
        # Add loan term as query parameter if API supports it
        params = {}
        # Some APIs accept loan_term as parameter, try it
        # If not supported, we'll filter results
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # API Ninjas typically returns an array of rate objects
        # Each object may have: term, rate, apr, etc.
        if isinstance(data, list) and len(data) > 0:
            # Look for rate matching loan term
            for rate_obj in data:
                # Try different possible field names for term
                term = None
                if 'loan_term' in rate_obj:
                    term = rate_obj['loan_term']
                elif 'term' in rate_obj:
                    term = rate_obj['term']
                elif 'years' in rate_obj:
                    term = rate_obj['years']
                elif 'loan_years' in rate_obj:
                    term = rate_obj['loan_years']
                
                # Check if term matches
                if term:
                    try:
                        term_int = int(term)
                        if term_int == loan_term:
                            # Found matching term, extract rate
                            rate = rate_obj.get('rate') or rate_obj.get('interest_rate') or rate_obj.get('apr')
                            if rate:
                                # Rate might already be a decimal or percentage
                                rate_float = float(rate)
                                # If rate > 1, assume it's a percentage, convert to decimal
                                if rate_float > 1:
                                    return rate_float / 100
                                return rate_float
                    except (ValueError, TypeError):
                        continue
            
            # If no exact match found, try to use first result or find closest
            # Some APIs return rates in order: 30yr, 15yr, etc.
            first_rate = data[0].get('rate') or data[0].get('interest_rate') or data[0].get('apr')
            if first_rate:
                rate_float = float(first_rate)
                if rate_float > 1:
                    return rate_float / 100
                return rate_float
        
        # Alternative: if API returns a dict
        if isinstance(data, dict):
            # Try different key formats
            for key_format in [f"{loan_term}year", f"{loan_term}_year", f"{loan_term}-year", str(loan_term)]:
                if key_format in data:
                    rate_obj = data[key_format]
                    if isinstance(rate_obj, dict):
                        rate = rate_obj.get('rate') or rate_obj.get('interest_rate')
                    else:
                        rate = rate_obj
                    if rate:
                        rate_float = float(rate)
                        if rate_float > 1:
                            return rate_float / 100
                        return rate_float
        
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.warning("⚠️ API Ninjas authentication failed. Using manual interest rate.")
        elif e.response.status_code == 403:
            st.warning("⚠️ API Ninjas access denied. Check your API key.")
        elif e.response.status_code == 404:
            # Mortgage rate endpoint might not be available - silently fail
            # st.caption("ℹ️ Mortgage rate API not available. Using manual input.")
            return None
        elif e.response.status_code == 429:
            st.warning("⚠️ API Ninjas rate limit exceeded. Using manual interest rate.")
        else:
            # Only show error for non-404 errors
            st.warning(f"⚠️ API Ninjas mortgage rate request failed ({e.response.status_code}). Using manual interest rate.")
        return None
    except Exception as e:
        # Silently fail - mortgage rate API might not be available
        return None

# RentCast API - Property Value Estimate
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_rentcast_property_value(address, api_key, bedrooms=None, bathrooms=None, square_footage=None, property_type=None):
    """
    Get property value estimate from RentCast API.
    Documentation: https://developers.rentcast.io/reference/property-valuation
    
    Args:
        address: Full property address (e.g., "123 Main St, Los Angeles, CA 90001")
        api_key: RentCast API key
        bedrooms: Number of bedrooms (optional, improves accuracy)
        bathrooms: Number of bathrooms (optional, improves accuracy)
        square_footage: Square footage (optional, improves accuracy)
        property_type: Property type (optional, improves accuracy)
    
    Returns:
        dict: Property value data including 'value' and 'comparables', or None if failed
    """
    if not api_key or not address:
        return None
    
    try:
        url = "https://api.rentcast.io/v1/avm/value"
        headers = {'X-Api-Key': api_key}
        params = {'address': address}
        
        # Add optional parameters to improve accuracy
        if bedrooms is not None:
            params['bedrooms'] = int(bedrooms)
        if bathrooms is not None:
            params['bathrooms'] = float(bathrooms)
        if square_footage is not None:
            params['squareFootage'] = int(square_footage)
        if property_type is not None:
            # Map property_category to RentCast property types
            # RentCast expects: "Single Family", "Condo", "Townhouse", "Multi-Family", etc.
            property_type_mapping = {
                'Entire House': 'Single Family',
                'Entire Apartment/Condo': 'Condo',
                'Private Room': 'Single Family',  # Default to Single Family
                'Hotel/Resort': 'Multi-Family'
            }
            mapped_type = property_type_mapping.get(property_type, 'Single Family')
            params['propertyType'] = mapped_type
        
        # Set lookupSubjectAttributes to false if we're providing our own attributes
        if bedrooms is not None or bathrooms is not None or square_footage is not None:
            params['lookupSubjectAttributes'] = 'false'
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.warning("⚠️ RentCast authentication failed. Check your API key.")
        elif e.response.status_code == 404:
            st.warning("⚠️ Property not found in RentCast database.")
        return None
    except Exception as e:
        return None

# HomeHarvest - Get Property Listing/Purchase Price (FREE)
# Note: Removed caching to ensure fresh results for each query
def get_homeharvest_property_price(address=None, latitude=None, longitude=None):
    """
    Get property listing/purchase price using HomeHarvest (free scraper).
    Uses the new scrape_property() API which returns a DataFrame.
    Searches Zillow, Redfin, and Realtor.com for property data.
    
    Note: HomeHarvest's scrape_property() only accepts a location string parameter,
    not separate latitude/longitude. Coordinates are ignored.
    
    Args:
        address: Full property address (e.g., "123 Main St, Los Angeles, CA 90001")
                 Can also be zip code, city, "city, state", etc.
                 This is the only parameter used by HomeHarvest.
        latitude: (Deprecated) Not used - HomeHarvest doesn't accept coordinates
        longitude: (Deprecated) Not used - HomeHarvest doesn't accept coordinates
    
    Returns:
        dict: Property data including 'list_price', 'estimated_value', 'sold_price', or None if failed
    """
    
    def strip_apartment_number(addr):
        """
        Remove apartment/unit numbers from address to improve search results.
        Examples:
        - "1155 S Grand Avenue, Apt 1706, Los Angeles, CA" -> "1155 S Grand Avenue, Los Angeles, CA"
        - "123 Main St Unit 5, City, State" -> "123 Main St, City, State"
        """
        if not addr:
            return addr
        
        import re
        # Patterns to match apartment/unit numbers
        patterns = [
            r',\s*Apt\.?\s*\d+[A-Z]?',  # ", Apt 1706" or ", Apt. 1706"
            r',\s*Apartment\s*\d+[A-Z]?',  # ", Apartment 1706"
            r',\s*Unit\s*\d+[A-Z]?',  # ", Unit 5"
            r',\s*#\s*\d+[A-Z]?',  # ", #5"
            r',\s*Suite\s*\d+[A-Z]?',  # ", Suite 5"
            r'\s+Apt\.?\s*\d+[A-Z]?',  # " Apt 1706" (no comma)
            r'\s+Unit\s*\d+[A-Z]?',  # " Unit 5" (no comma)
        ]
        
        cleaned = addr
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up any double commas or extra spaces
        cleaned = re.sub(r',\s*,', ',', cleaned)  # Remove double commas
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Remove extra spaces
        cleaned = cleaned.strip()
        
        return cleaned
    # #region agent log
    import sys
    import json
    try:
        with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "streamlit_app.py:387",
                "message": "Function entry - checking Python version",
                "data": {
                    "python_version": sys.version,
                    "python_version_info": list(sys.version_info),
                    "address": address,
                    "latitude": latitude,
                    "longitude": longitude
                },
                "timestamp": int(__import__('time').time() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    try:
        # Import homeharvest - will fail silently on Python 3.9 due to typing syntax
        try:
            # #region agent log
            try:
                with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "streamlit_app.py:405",
                        "message": "Attempting HomeHarvest import",
                        "data": {},
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            
            from homeharvest import scrape_property
            
            # #region agent log
            try:
                with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "streamlit_app.py:405",
                        "message": "HomeHarvest import successful",
                        "data": {},
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            
        except (TypeError, ImportError, Exception) as import_error:
            # #region agent log
            try:
                with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "streamlit_app.py:406",
                        "message": "HomeHarvest import failed",
                        "data": {
                            "error_type": str(type(import_error)),
                            "error_message": str(import_error),
                            "error_repr": repr(import_error),
                            "python_version": sys.version
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            
            # HomeHarvest requires Python 3.10+ due to typing syntax (list[dict] | None)
            error_msg = str(import_error)
            if "GenericAlias" in error_msg or "type annotation" in error_msg.lower():
                st.warning(f"⚠️ HomeHarvest requires Python 3.10+. Your Python version may be incompatible. Error: {error_msg[:200]}")
            elif "No module named" in error_msg or "ImportError" in str(type(import_error)):
                st.warning(f"⚠️ HomeHarvest not installed. Run: pip install homeharvest")
            else:
                st.warning(f"⚠️ HomeHarvest import failed: {error_msg[:200]}")
            return None
        
        # Helper function to search and extract property data
        # Note: Removed listing_type parameter - searching all types like Colab
        def search_and_extract(location, use_radius=False):
            """Search for properties and extract data using Property Schema field names"""
            # #region agent log
            try:
                import json
                with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "B",
                        "location": "streamlit_app.py:492",
                        "message": "search_and_extract called",
                        "data": {
                            "location": location,
                            "use_radius": use_radius,
                            "note": "No listing_type filter - searching all types like Colab"
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            
            # Track if we encounter 403 errors
            encountered_403 = False
            
            try:
                import time
                
                # Test: Try calling exactly like Colab does first (simple, direct call)
                # Colab example: scrape_property(location="San Diego, CA", listing_type="sold", past_days=30)
                
                # Strip apartment numbers from address (apartment numbers can break scraping)
                cleaned_location = strip_apartment_number(location) if location else location
                
                # #region agent log
                try:
                    import json
                    with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "D",
                            "location": "streamlit_app.py:555",
                            "message": "Stripping apartment number from address",
                            "data": {
                                "original_location": location,
                                "cleaned_location": cleaned_location
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + '\n')
                except: pass
                # #endregion
                
                # Prepare parameters - start simple like Colab (no beds/baths/sqft filters)
                scrape_params = {
                    'location': cleaned_location,  # Use cleaned address without apartment number
                    'past_days': 365  # Match Colab example exactly
                }
                
                # Only add radius if explicitly requested (might trigger different endpoints)
                if use_radius and address:
                    scrape_params['radius'] = 0.5
                
                # Don't use limit initially - let's see if that's causing issues
                # scrape_params['limit'] = 10  # Commented out to match Colab more closely
                
                # #region agent log
                try:
                    import json
                    with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                            "location": "streamlit_app.py:510",
                            "message": "Calling scrape_property DIRECTLY (no ThreadPoolExecutor) - matching Colab approach",
                            "data": {
                                "scrape_params": scrape_params,
                                "use_threading": False,
                                "note": "Testing direct call like Colab to see if ThreadPoolExecutor is the issue"
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + '\n')
                except: pass
                # #endregion
                
                # Try direct call first (like Colab) - no ThreadPoolExecutor
                start_time = time.time()
                try:
                    properties_df = scrape_property(**scrape_params)
                    elapsed = time.time() - start_time
                    
                    # #region agent log
                    try:
                        import json
                        with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                                "location": "streamlit_app.py:550",
                                "message": "Direct scrape_property call completed",
                                "data": {
                                    "elapsed_seconds": elapsed,
                                    "result_type": str(type(properties_df)),
                                    "result_length": len(properties_df) if hasattr(properties_df, '__len__') else None,
                                    "success": True
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + '\n')
                    except: pass
                    # #endregion
                    
                except Exception as direct_error:
                    elapsed = time.time() - start_time
                    error_msg = str(direct_error)
                    
                    # #region agent log
                    try:
                        import json
                        import traceback
                        with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                                "location": "streamlit_app.py:570",
                                "message": "Direct scrape_property call failed",
                                "data": {
                                    "elapsed_seconds": elapsed,
                                    "error_type": str(type(direct_error)),
                                    "error_message": error_msg,
                                    "traceback": traceback.format_exc(),
                                    "success": False
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + '\n')
                    except: pass
                    # #endregion
                    
                    # Check if it's a 403 error
                    if "403" in error_msg or "Forbidden" in error_msg or "RetryError" in str(type(direct_error)):
                        raise Exception("403_FORBIDDEN_ERROR") from direct_error
                    # Re-raise other exceptions
                    raise
                
                # #region agent log
                try:
                    import json
                    with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "B",
                            "location": "streamlit_app.py:550",
                            "message": "Processing results",
                            "data": {
                                "df_is_none": properties_df is None,
                                "df_length": len(properties_df) if properties_df is not None else None,
                                "df_columns": list(properties_df.columns) if properties_df is not None and hasattr(properties_df, 'columns') else None
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + '\n')
                except: pass
                # #endregion
                
                if properties_df is not None and len(properties_df) > 0:
                    # Get the first property (closest match)
                    property_data = properties_df.iloc[0].to_dict()
                    
                    # #region agent log
                    try:
                        import json
                        with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "B",
                                "location": "streamlit_app.py:565",
                                "message": "Extracting property data",
                                "data": {
                                    "available_keys": list(property_data.keys()),
                                    "has_list_price": 'list_price' in property_data,
                                    "has_price": 'price' in property_data,
                                    "list_price_value": property_data.get('list_price'),
                                    "price_value": property_data.get('price')
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + '\n')
                    except: pass
                    # #endregion
                    
                    # Extract using Property Schema field names (from GitHub documentation)
                    # According to docs: list_price is the correct field name, not 'price'
                    return {
                        'list_price': property_data.get('list_price'),  # Correct field name per GitHub docs
                        'estimated_value': property_data.get('tax_assessed_value') or property_data.get('estimated_value'),
                        'sold_price': property_data.get('sold_price') or property_data.get('last_sold_price'),
                        'last_sold_price': property_data.get('last_sold_price') or property_data.get('sold_price'),
                        'last_sold_date': property_data.get('last_sold_date') or property_data.get('sold_date'),
                        'property_id': property_data.get('property_id') or property_data.get('listing_id'),
                        'property_url': property_data.get('property_url') or property_data.get('permalink'),
                        'beds': property_data.get('beds'),
                        'baths': property_data.get('full_baths'),
                        'sqft': property_data.get('sqft'),
                        'year_built': property_data.get('year_built'),
                        'address': property_data.get('formatted_address') or property_data.get('address') or address or location,
                        'status': property_data.get('status') or property_data.get('mls_status'),
                        'mls': property_data.get('mls'),
                        'mls_id': property_data.get('mls_id')
                    }
            except Exception as e:
                # #region agent log
                try:
                    import json
                    import traceback
                    with open('/Users/kevinhoung/AirBnB Project/us_rental_calculator/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "B",
                            "location": "streamlit_app.py:590",
                            "message": "Exception in search_and_extract",
                            "data": {
                                "error_type": str(type(e)),
                                "error_message": str(e),
                                "traceback": traceback.format_exc()
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + '\n')
                except: pass
                # #endregion
                
                # Check if it's a 403 Forbidden error (Realtor.com blocking)
                error_msg = str(e)
                if "403" in error_msg or "Forbidden" in error_msg or "RetryError" in str(type(e)):
                    # Raise a special exception so outer function can detect it
                    raise Exception("403_FORBIDDEN_ERROR") from e
                
                return None
            return None
        
        # Strategy: Search without listing_type filter (like Colab) - gets all types
        # HomeHarvest only accepts location string, not lat/lon coordinates
        # Strip apartment number from address first (apartment numbers break scraping)
        if not address:
            return None
        
        cleaned_address = strip_apartment_number(address)
        encountered_403 = False
        
        # Try with radius first (better for specific addresses per GitHub docs)
        try:
            result = search_and_extract(cleaned_address, use_radius=True)
            if result:
                return result
        except Exception as e:
            error_msg = str(e)
            if "403_FORBIDDEN_ERROR" in error_msg or "403" in error_msg or "Forbidden" in error_msg:
                encountered_403 = True
            # Continue to try without radius
        
        # Fallback: try without radius
        try:
            result = search_and_extract(cleaned_address, use_radius=False)
            if result:
                return result
        except Exception as e:
            error_msg = str(e)
            if "403_FORBIDDEN_ERROR" in error_msg or "403" in error_msg or "Forbidden" in error_msg:
                encountered_403 = True
        
        # Return special indicator if we encountered 403 errors
        if encountered_403:
            return {'_error_403': True}
        
        return None
    except ImportError:
        st.warning("⚠️ HomeHarvest not installed. Run: pip install homeharvest")
        return None
    except Exception as e:
        # Show error for debugging
        error_msg = str(e)
        import traceback
        error_trace = traceback.format_exc()
        
        if "GenericAlias" in error_msg or "type annotation" in error_msg.lower():
            st.warning(f"⚠️ HomeHarvest typing error (Python 3.9 compatibility issue): {error_msg[:200]}")
            st.caption("💡 Tip: HomeHarvest works best with Python 3.10+. Using RentCast fallback.")
        else:
            st.warning(f"⚠️ HomeHarvest error: {error_msg[:200]}")
            # Show full traceback in expander for debugging
            with st.expander("🔍 See full error details"):
                st.code(error_trace)
        return None

# RentCast API - Rental Comparables
# Note: Removed caching to ensure fresh results and better error visibility
def get_rentcast_rental_comparables(address, api_key, bedrooms=None, bathrooms=None, square_footage=None, property_type=None):
    """
    Get rental comparables from RentCast API.
    Documentation: https://developers.rentcast.io/reference/property-valuation
    
    Args:
        address: Full property address
        api_key: RentCast API key
        bedrooms: Number of bedrooms (optional, improves accuracy)
        bathrooms: Number of bathrooms (optional, improves accuracy)
        square_footage: Square footage (optional, improves accuracy)
        property_type: Property type (optional, improves accuracy)
    
    Returns:
        dict: Rental data including 'rent' and 'comparables' list, or None if failed
    """
    if not api_key or not address:
        return None
    
    try:
        url = "https://api.rentcast.io/v1/avm/rent/long-term"
        headers = {'X-Api-Key': api_key}
        params = {'address': address}
        
        # Add optional parameters to improve accuracy
        if bedrooms is not None:
            params['bedrooms'] = int(bedrooms)
        if bathrooms is not None:
            params['bathrooms'] = float(bathrooms)
        if square_footage is not None:
            params['squareFootage'] = int(square_footage)
        if property_type is not None:
            # Map property_category to RentCast property types
            property_type_mapping = {
                'Entire House': 'Single Family',
                'Entire Apartment/Condo': 'Condo',
                'Private Room': 'Single Family',
                'Hotel/Resort': 'Multi-Family'
            }
            mapped_type = property_type_mapping.get(property_type, 'Single Family')
            params['propertyType'] = mapped_type
        
        # Set lookupSubjectAttributes to false if we're providing our own attributes
        if bedrooms is not None or bathrooms is not None or square_footage is not None:
            params['lookupSubjectAttributes'] = 'false'
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        # RentCast API response received
        
        return data
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP {e.response.status_code}"
        try:
            error_detail = e.response.json()
            error_msg += f": {error_detail}"
        except:
            error_msg += f": {e.response.text[:200]}"
        
        if e.response.status_code == 401:
            st.warning("⚠️ RentCast authentication failed for rental comparables. Check your API key.")
        elif e.response.status_code == 404:
            st.warning("⚠️ Property not found in RentCast database for rental data.")
        else:
            st.warning(f"⚠️ RentCast API error: {error_msg}")
        return None
    except Exception as e:
        st.warning(f"⚠️ RentCast API exception: {str(e)}")
        return None

# Property Value Estimation API function (fallback)
# Note: API Ninjas doesn't have a property value API, so we'll use property tax data to estimate
@st.cache_data(ttl=86400)  # Cache for 24 hours
def estimate_property_value_from_tax(city, state_abbr, api_key):
    """
    Estimate property value using property tax data.
    Uses median property tax rate and typical annual property tax amounts to estimate value.
    
    Args:
        city: City name
        state_abbr: State abbreviation
        api_key: API Ninjas API key
    
    Returns:
        float: Estimated property value, or None if failed
    """
    if not api_key or not city or not state_abbr:
        return None
    
    try:
        # Get property tax data
        url = "https://api.api-ninjas.com/v1/propertytax"
        headers = {'X-Api-Key': api_key}
        params = {'city': city, 'state': state_abbr.upper()}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            # Get median property tax rate (50th percentile)
            tax_rate = data[0].get('property_tax_50th_percentile')
            if tax_rate:
                tax_rate_float = float(tax_rate)
                
                # Estimate property value using typical annual property tax
                # National average annual property tax varies by state:
                # - High tax states (NJ, IL, NH): $5,000-8,000
                # - Medium tax states (CA, NY, TX): $3,000-5,000
                # - Low tax states (AL, LA, HI): $1,000-2,500
                # We'll use a conservative estimate based on the tax rate
                # Higher tax rate usually means higher property values
                
                # Estimate annual tax based on tax rate
                # If tax rate is > 1.5%, assume higher value area
                if tax_rate_float > 0.015:
                    estimated_annual_tax = 6000  # Higher value area
                elif tax_rate_float > 0.01:
                    estimated_annual_tax = 4000  # Medium value area
                else:
                    estimated_annual_tax = 2500  # Lower value area
                
                # Calculate estimated value: value = annual_tax / tax_rate
                estimated_value = estimated_annual_tax / tax_rate_float
                return estimated_value
        
        return None
    except Exception:
        return None

# API Ninjas Property Tax API function
@st.cache_data(ttl=86400)  # Cache for 24 hours (property tax rates don't change often)
def get_apininjas_property_tax(city, state_abbr, api_key):
    """
    Fetch property tax rate from API Ninjas.
    Documentation: https://api-ninjas.com/api/propertytax
    
    Args:
        city: City name (e.g., "Los Angeles")
        state_abbr: State abbreviation (e.g., "CA")
        api_key: API Ninjas API key
    
    Returns:
        float: Property tax rate as decimal (e.g., 0.012 for 1.2%), or None if failed
    """
    if not api_key or not city or not state_abbr:
        return None
    
    try:
        url = "https://api.api-ninjas.com/v1/propertytax"
        headers = {
            'X-Api-Key': api_key
        }
        params = {
            'city': city,
            'state': state_abbr.upper()
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # API returns array of regions, use 75th percentile rate (more conservative estimate)
        if isinstance(data, list) and len(data) > 0:
            # Use the first result's 75th percentile property tax rate
            tax_rate = data[0].get('property_tax_75th_percentile')
            if tax_rate:
                return float(tax_rate)  # Already a decimal (e.g., 0.012 for 1.2%)
        else:
            # Debug: Show what we got back
            st.caption(f"⚠️ Property tax API returned empty results for {city}, {state_abbr}")
        
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.warning("⚠️ API Ninjas authentication failed for property tax. Check your API key.")
        elif e.response.status_code == 403:
            st.warning("⚠️ API Ninjas access denied for property tax. Check your API key permissions.")
        elif e.response.status_code == 429:
            st.warning("⚠️ API Ninjas rate limit exceeded for property tax.")
        else:
            error_text = e.response.text[:200] if hasattr(e.response, 'text') else str(e)
            st.warning(f"⚠️ API Ninjas property tax request failed ({e.response.status_code}): {error_text}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error fetching property tax rate: {str(e)}")
        return None

# API Ninjas Mortgage Calculator API function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_apininjas_mortgage_calculation(loan_amount, interest_rate, duration_years, annual_property_tax, annual_home_insurance, api_key):
    """
    Calculate mortgage payment using API Ninjas Mortgage Calculator.
    Documentation: https://api-ninjas.com/api/mortgagecalculator
    
    Args:
        loan_amount: Principal loan amount
        interest_rate: Annual interest rate as decimal (e.g., 0.065 for 6.5%)
        duration_years: Loan term in years
        annual_property_tax: Annual property tax amount
        annual_home_insurance: Annual home insurance amount
        api_key: API Ninjas API key
    
    Returns:
        dict: Contains monthly_payment, annual_payment, total_interest_paid, or None if failed
    """
    if not api_key:
        return None
    
    try:
        url = "https://api.api-ninjas.com/v1/mortgagecalculator"
        headers = {
            'X-Api-Key': api_key
        }
        params = {
            'loan_amount': int(loan_amount),
            'interest_rate': interest_rate * 100,  # Convert to percentage
            'duration_years': duration_years,
            'annual_property_tax': annual_property_tax,
            'annual_home_insurance': annual_home_insurance
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.warning("⚠️ API Ninjas authentication failed for mortgage calculator. Check your API key.")
        elif e.response.status_code == 403:
            st.warning("⚠️ API Ninjas access denied for mortgage calculator. Check your API key permissions.")
        elif e.response.status_code == 429:
            st.warning("⚠️ API Ninjas rate limit exceeded for mortgage calculator.")
        else:
            error_text = e.response.text[:200] if hasattr(e.response, 'text') else str(e)
            st.warning(f"⚠️ API Ninjas mortgage calculator request failed ({e.response.status_code}): {error_text}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error fetching mortgage calculation: {str(e)}")
        return None

# Helper function to extract state abbreviation from city name
def extract_state_from_city(city_display):
    """
    Extract state abbreviation from city display name.
    Example: "Los Angeles, CA" -> "CA"
    """
    if ", " in city_display:
        return city_display.split(", ")[-1]
    return None

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
        # Load Decision Tree Model
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
        st.markdown("<h3 style='text-align: center; '>Physical Properties</h3>", unsafe_allow_html=True)
        accommodates = st.number_input("1. Accommodates (Guests)", min_value=1, value=2)
        bedrooms = st.number_input("2. Bedrooms", min_value=0, value=1)
        beds = st.number_input("3. Beds", min_value=1, value=1)
        
        # Updated: step=0.5 allows for 1.0, 1.5, 2.0, etc.
        bathrooms = st.number_input("4. Bathrooms", min_value=0.5, value=1.0, step=0.5)
        
        # Square footage slider (optional, for improved comparables accuracy)
        square_footage = st.slider("5. Square Footage (Optional)", 
                                  min_value=250, 
                                  max_value=3000, 
                                  value=1000, 
                                  step=50,
                                  help="Optional: Helps improve comparables accuracy. Leave at default if unknown.")
        
        st.markdown("<h3 style='text-align: center; '>Location</h3>", unsafe_allow_html=True)
        
        # --- NEW INSTRUCTION TEXT ---
        st.info("ℹ️ **Note:** This model is trained only on the 30 specific cities listed below.")
        
        # 1. Strict City Selection
        selected_city_display = st.selectbox("6a. Select City (Required)", ALLOWED_CITIES_DISPLAY)
        
        # 2. Optional Street Address
        street_address = st.text_input("6b. Street Address (Optional)", 
                                      placeholder="e.g., 100 Main Street")
        
    with col2:
        st.markdown("<h3 style='text-align: center; '>Quality & Booking</h3>", unsafe_allow_html=True)
        review_scores_cleanliness = st.slider("7. Cleanliness Score (1-5)", 1.0, 5.0, 4.0, step=0.25)
        review_scores_location = st.slider("8. Location Score (1-5)", 1.0, 5.0, 4.0, step=0.25)
        
        instant_bookable = st.selectbox("9. Instant Bookable", ["Yes", "No"])
        instant_bookable_nb = 1 if instant_bookable == "Yes" else 0 
        
        property_categories = [
            'Entire Apartment/Condo', 'Entire House', 'Private Room', 
            'Hotel/Resort' 
        ]
        property_category = st.selectbox("10. Property Type Category", property_categories)
        
        # Predict Price section directly below Quality & Booking
        st.markdown("<h3 style='text-align: center; '>Neural Pricing</h3>", unsafe_allow_html=True)
        
        # Center the button using columns - wider middle column for button
        button_col1, button_col2, button_col3 = st.columns([0.5, 3, 0.5])
        
        with button_col2:
            # Center the image using CSS
            st.markdown("""
            <style>
            .centered-image {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Show image centered
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image("images/buttonv2.png", width=200, use_container_width=False)
            
            # Button below the image - centered
            predict_button = st.button(
                "Predict Price", 
                use_container_width=True,
                type="primary"
            )
        
            # Loading animation placeholder (right under the button, centered)
            loading_placeholder = st.empty()
            
            # Add some spacing to move animation up
            st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- 4. PREDICTION LOGIC ---
    # Initialize session state for prediction results
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    
    # Check if button was clicked (run new prediction)
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
            
            # Get Mapbox token from secrets
            # Streamlit automatically loads .streamlit/secrets.toml when running locally
            try:
                mapbox_token = st.secrets.get('mapbox', {}).get('token') or get_secret('mapbox.token')
            except Exception:
                mapbox_token = get_secret('mapbox.token')
            
            # Debug: Check if token is loaded (only show if not found to avoid clutter)
            if not mapbox_token:
                st.error("❌ Mapbox token not found! Check your .streamlit/secrets.toml file.")
                st.info("💡 Make sure you have: `[mapbox]` section with `token = \"your-token\"`")
                st.info("💡 Also check: Are you running Streamlit from the project root directory?")
                st.info(f"💡 Current working directory: {os.getcwd()}")
                st.info(f"💡 Secrets file exists: {os.path.exists('.streamlit/secrets.toml')}")
            
            # Try geocoding with Mapbox
            geocoded_city = None
            geocoded_state = None
            
            if mapbox_token:
                # Try full address first (if street address provided)
                if street_address:
                    # Show what we're trying to geocode
                    result = geocode_with_mapbox(search_query, mapbox_token)
                    if result and result[0] is not None and result[1] is not None:
                        latitude, longitude, geocoded_city, geocoded_state = result
                        st.success(f"✅ Successfully geocoded: ({latitude:.6f}, {longitude:.6f})")
                    else:
                        # If full address failed, try just the city
                        st.warning(f"⚠️ Could not geocode full address '{search_query}'. Trying city only...")
                        result = geocode_with_mapbox(search_city_value, mapbox_token)
                        if result and result[0] is not None and result[1] is not None:
                            latitude, longitude, geocoded_city, geocoded_state = result
                            st.info(f"📍 Using city center coordinates for {selected_city_display}")
                        else:
                            st.warning(f"⚠️ Could not geocode city '{search_city_value}'. Using fallback coordinates.")
                else:
                    # No street address, just geocode the city
                    st.caption(f"🔍 Geocoding city: '{search_city_value}'")
                    result = geocode_with_mapbox(search_city_value, mapbox_token)
                    if result and result[0] is not None and result[1] is not None:
                        latitude, longitude, geocoded_city, geocoded_state = result
                        st.info(f"📍 Geocoded city: {selected_city_display}")
                    else:
                        st.warning(f"⚠️ Could not geocode city '{search_city_value}'. Using fallback coordinates.")
                
                # If still no result, use fallback coordinates
                if latitude is None or longitude is None:
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
                        st.caption(f"ℹ️ Using city center coordinates for {selected_city_display}")
                    else:
                        st.error("❌ Could not determine coordinates. Please try again.")
                        st.stop()
            else:
                # No Mapbox token - use fallback coordinates
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
                    st.caption(f"ℹ️ Using city center coordinates for {selected_city_display}")
                else:
                    st.error("❌ Could not determine coordinates. Please configure Mapbox token in secrets.")
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
            
            # Query nearby listings
            with st.spinner('Loading nearby competitor listings...'):
                competitor_df = query_nearby_listings(latitude, longitude, radius_km=5, limit=200)
            
            # Store results in session state for persistence
            # Clear any old rental comparables when new prediction is made
            if 'rental_comparables' in st.session_state:
                del st.session_state['rental_comparables']
            
            st.session_state.prediction_results = {
                'price_prediction': price_prediction,
                'latitude': latitude,
                'longitude': longitude,
                'search_query': search_query,
                'competitor_df': competitor_df,
                'city': geocoded_city,  # City from geocoding (for property tax lookup)
                'state': geocoded_state,  # State from geocoding (for property tax lookup)
                'bedrooms': bedrooms,  # Store for API calls
                'bathrooms': bathrooms,  # Store for API calls
                'square_footage': square_footage,  # Store for API calls
                'property_category': property_category,  # Store for API calls
            }
            
            # Fetch rental comparables right after prediction (not waiting for ROI calculator)
            rentcast_api_key = get_secret('rentcast.api_key')
            if not rentcast_api_key:
                pass  # RentCast API key not found
            elif not search_query:
                pass  # No search_query available
            else:
                with st.spinner('Fetching rental comparables from RentCast...'):
                    rental_comparables_data = get_rentcast_rental_comparables(
                        search_query, 
                        rentcast_api_key,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        square_footage=square_footage,
                        property_type=property_category
                    )
                    # Store in session state for use later
                    if rental_comparables_data:
                        st.session_state.rental_comparables = rental_comparables_data
                        # Debug: Show if comparables were returned
                        comparables_count = len(rental_comparables_data.get('comparables', []))
                        if comparables_count > 0:
                            st.success(f"✅ Found {comparables_count} comparable rental properties")
                        else:
                            st.info("ℹ️ RentCast returned data but no comparables found. Check debug info below.")
                            # Show what we got back
                            st.caption(f"🔍 RentCast response keys: {list(rental_comparables_data.keys())}")
                    else:
                        st.session_state.rental_comparables = None
                        st.warning("⚠️ RentCast API returned None. Check error messages above.")
            
            # Debug: Show the coordinates being used
        
        except Exception as e:
            st.error("Prediction Logic Failed.")
            st.exception(e)
            st.session_state.prediction_results = None
    
    # Display results if we have them (either from new prediction or stored)
    if st.session_state.prediction_results is not None:
        # Load stored results
        price_prediction = st.session_state.prediction_results['price_prediction']
        latitude = st.session_state.prediction_results['latitude']
        longitude = st.session_state.prediction_results['longitude']
        search_query = st.session_state.prediction_results['search_query']
        competitor_df = st.session_state.prediction_results['competitor_df']
        
        # Display location and price on one line
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.info(f"📍 Locating: **{search_query}**")
        with info_col2:
            st.success(f"֎ Estimated Price: **${price_prediction:,.2f}** per night")
        
        # --- MAP WITH COMPETITOR LISTINGS ---
        st.markdown("---")
        
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
        
        # --- ROI CALCULATOR SECTION ---
        # Always show ROI calculator if we have prediction results
        st.markdown("---")
        st.subheader("🏦 Investment Analysis")
        
        with st.expander("🧮 Investment Calculator", expanded=True):
            # Initialize session state for ROI calculator values
            if 'roi_calculator_values' not in st.session_state:
                st.session_state.roi_calculator_values = {
                    'listing_price': None,
                    'down_payment_pct': 0.20,
                    'interest_rate': 0.065,
                    'loan_term': 30,
                    'avg_nightly_rate': round(price_prediction, 2),
                    'occupancy_rate': 0.60,
                    'property_tax_rate': 0.012,
                    'insurance_monthly': 150,
                    'utilities_monthly': 250,
                    'management_fee_pct': 0.10
                }
            
            # Get API keys (used in multiple places)
            # Uses local .streamlit/secrets.toml when running locally, Streamlit secrets in production
            try:
                api_ninjas_key = st.secrets.get('api_ninjas', {}).get('api_key') or get_secret('api_ninjas.api_key')
            except Exception:
                api_ninjas_key = get_secret('api_ninjas.api_key')
            
            try:
                rentcast_api_key = st.secrets.get('rentcast', {}).get('api_key') or get_secret('rentcast.api_key')
            except Exception:
                rentcast_api_key = get_secret('rentcast.api_key')
            
            # Debug: Check if API keys were found (only show if not found to avoid clutter)
            if not api_ninjas_key:
                st.warning("⚠️ API Ninjas key not found in secrets. Check your .streamlit/secrets.toml file.")
                st.info("💡 Make sure you have: `[api_ninjas]` section with `api_key = \"your-key\"`")
            
            # Check RentCast API key
            if not rentcast_api_key:
                st.warning("⚠️ RentCast API key not found in secrets. Check your .streamlit/secrets.toml file.")
                st.info("💡 Make sure you have: `[rentcast]` section with `api_key = \"your-key\"`")
            else:
                # Silently confirm RentCast is configured (don't show message to avoid clutter)
                pass
            
            # Create three columns for inputs (no form - automatic updates)
            inv_col1, inv_col2, inv_col3 = st.columns(3)
            
            with inv_col1: 
                st.markdown("#### 1. Purchase Details")
                
                # Try to fetch property value from RentCast API (preferred) or fallback to estimation
                estimated_property_value = None
                rentcast_data = None
                homeharvest_data = None
                rental_comparables_data = None
                price_source_name = None  # Track which source actually provided the price
                
                if st.session_state.prediction_results:
                    search_query = st.session_state.prediction_results.get('search_query', '')
                    latitude = st.session_state.prediction_results.get('latitude')
                    longitude = st.session_state.prediction_results.get('longitude')
                    
                    # HomeHarvest temporarily disabled - API is currently broken
                    # Keeping code below for future use when HomeHarvest is fixed
                    
                    # Try HomeHarvest first (FREE - no API key needed)
                    # HomeHarvest only accepts location string (address), not lat/lon coordinates
                    if search_query:
                        with st.spinner('Fetching property listing price from HomeHarvest...'):
                            st.caption(f"🔍 Searching HomeHarvest with address: '{search_query}'")
                            
                            homeharvest_data = get_homeharvest_property_price(
                                address=search_query
                            )
                            
                            # Debug: Show what we got back
                            if homeharvest_data:
                                # Check if this is a 403 error indicator
                                if homeharvest_data.get('_error_403'):
                                    st.info("ℹ️ HomeHarvest unavailable: Realtor.com is rate-limiting your local IP address (429/403).")
                                    st.caption("💡 **Why it works in Google Colab but not locally:**")
                                    st.caption("   • Your local IP has been rate-limited from too many requests (429 Too Many Requests)")
                                    st.caption("   • Google Colab uses different IP addresses that aren't rate-limited")
                                    st.caption("   • This happens when making many requests during testing/debugging")
                                    st.caption("")
                                    st.caption("**Solutions:**")
                                    st.caption("   1. ⏰ **Wait 15-60 minutes** for rate limit to reset (easiest)")
                                    st.caption("   2. 🌐 **Use a different network** (mobile hotspot, different WiFi, VPN)")
                                    st.caption("   3. 💰 **Use RentCast API** (recommended for production - no rate limits)")
                                    st.caption("   4. ☁️ **Run from Google Colab** or a cloud server (different IP)")
                                else:
                                    st.caption(f"🔍 HomeHarvest returned data with keys: {list(homeharvest_data.keys())}")
                                    # Prefer list_price (current listing), then estimated_value, then sold_price
                                    homeharvest_price = (
                                        homeharvest_data.get('list_price') or 
                                        homeharvest_data.get('estimated_value') or 
                                        homeharvest_data.get('sold_price') or 
                                        homeharvest_data.get('last_sold_price')
                                    )
                                    if homeharvest_price:
                                        estimated_property_value = homeharvest_price
                                        price_source_name = "HomeHarvest"
                                        price_source = "current listing" if homeharvest_data.get('list_price') else "estimated value" if homeharvest_data.get('estimated_value') else "last sold price"
                                        st.success(f"✅ Found property price from HomeHarvest: ${estimated_property_value:,.0f}")
                                    else:
                                        st.warning("⚠️ HomeHarvest found property but no price data. Trying RentCast...")
                                        st.caption(f"🔍 Debug - Available fields: {homeharvest_data}")
                            else:
                                st.info("ℹ️ HomeHarvest returned no data. Trying RentCast...")
                                st.caption("💡 This could mean: timeout occurred, no properties found, or Realtor.com is blocking requests.")
                    
                    # Fallback to RentCast API if HomeHarvest didn't find a price
                    if not estimated_property_value:
                        rentcast_api_key = get_secret('rentcast.api_key')
                        if rentcast_api_key and search_query:
                            # Get property attributes from session state for improved accuracy
                            pred_bedrooms = st.session_state.prediction_results.get('bedrooms')
                            pred_bathrooms = st.session_state.prediction_results.get('bathrooms')
                            pred_sqft = st.session_state.prediction_results.get('square_footage')
                            pred_property_type = st.session_state.prediction_results.get('property_category')
                            
                            with st.spinner('Fetching property value from RentCast...'):
                                rentcast_data = get_rentcast_property_value(
                                    search_query, 
                                    rentcast_api_key,
                                    bedrooms=pred_bedrooms,
                                    bathrooms=pred_bathrooms,
                                    square_footage=pred_sqft,
                                    property_type=pred_property_type
                                )
                                if rentcast_data and 'value' in rentcast_data:
                                    estimated_property_value = rentcast_data.get('value')
                                    price_source_name = "RentCast"
                                    st.success(f"✅ Found property value from RentCast: ${estimated_property_value:,.0f}")
                                elif rentcast_data is None:
                                    st.warning("⚠️ RentCast API returned no data. Check your API key or property address.")
                    
                    # Fallback to property tax estimation if both fail
                    if not estimated_property_value and api_ninjas_key:
                        city = st.session_state.prediction_results.get('city')
                        state_abbr = st.session_state.prediction_results.get('state')
                        
                        # If geocoding didn't provide city/state, try to extract from search_query
                        if not city or not state_abbr:
                            if ", " in search_query:
                                parts = search_query.split(", ")
                                if len(parts) >= 2:
                                    potential_state = parts[-1].strip()
                                    if len(parts) >= 3:
                                        city = parts[-2].strip()
                                    else:
                                        city = parts[0].strip()
                                    if len(potential_state) == 2 and potential_state.isalpha():
                                        state_abbr = potential_state.upper()
                        
                        if city and state_abbr:
                            with st.spinner('Estimating property value from tax data...'):
                                estimated_property_value = estimate_property_value_from_tax(city, state_abbr, api_ninjas_key)
                                if estimated_property_value:
                                    price_source_name = "Property Tax Data"
                    
                    # Note: Rental comparables are now fetched right after prediction is made
                    # (moved to prediction section for better UX)
                else:
                    rental_comparables_data = None
                    st.session_state.rental_comparables = None
                
                # Use text input for dollar formatting without +/- buttons
                # Pre-fill with estimated value if available
                default_value = ""
                if estimated_property_value:
                    default_value = f"{int(estimated_property_value):,}"
                    # Show the correct source based on which one actually provided the value
                    if price_source_name == "HomeHarvest":
                        price_source_detail = "current listing" if homeharvest_data and homeharvest_data.get('list_price') else "estimated value" if homeharvest_data and homeharvest_data.get('estimated_value') else "last sold price"
                        st.caption(f"💡 Property price: ${int(estimated_property_value):,} (from HomeHarvest - {price_source_detail})")
                    elif price_source_name == "RentCast":
                        st.caption(f"💡 Property value: ${int(estimated_property_value):,} (from RentCast)")
                    elif price_source_name == "Property Tax Data":
                        st.caption(f"💡 Estimated property value: ${int(estimated_property_value):,} (based on local property tax data)")
                    else:
                        # Fallback if source wasn't tracked
                        st.caption(f"💡 Property value: ${int(estimated_property_value):,}")
                
                listing_price_input = st.text_input(
                    "Property Purchase Price ($)", 
                    value=default_value, 
                    placeholder="100,000", 
                    help="Enter amount with or without commas (e.g., 100000 or 100,000). Value pre-filled from RentCast if available."
                )
                
                # Parse the input: remove commas, dollar signs, and whitespace
                listing_price = None
                if listing_price_input:
                    # Remove $, commas, and whitespace
                    cleaned_input = listing_price_input.replace("$", "").replace(",", "").replace(" ", "").strip()
                    try:
                        listing_price = float(cleaned_input) if cleaned_input else None
                    except ValueError:
                        listing_price = None
                
                # Use session state values as defaults
                default_down_payment = int(st.session_state.roi_calculator_values['down_payment_pct'] * 100)
                default_interest = st.session_state.roi_calculator_values['interest_rate'] * 100
                default_loan_term_idx = [10, 15, 20, 30].index(st.session_state.roi_calculator_values['loan_term']) if st.session_state.roi_calculator_values['loan_term'] in [10, 15, 20, 30] else 1
                
                down_payment_pct = st.slider("Down Payment (%)", 5, 50, default_down_payment) / 100
                loan_term = st.selectbox("Loan Term (Years)", [10, 15, 20, 30], index=default_loan_term_idx)
                
                # Get API Ninjas mortgage rate if available
                api_rate = None
                use_api_rate = False
                
                # Fetch API Ninjas rate if we have the key
                # Note: Mortgage rate API may not be available, so this is optional
                if api_ninjas_key:
                    with st.spinner('Fetching current mortgage rate...'):
                        api_rate = get_apininjas_mortgage_rate(
                            loan_term=loan_term,
                            api_key=api_ninjas_key
                        )
                    
                    if api_rate:
                        use_api_rate = st.checkbox(
                            f"Use current market rate ({api_rate*100:.2f}%)", 
                            value=True,
                            help="Current market rate from API Ninjas. Uncheck to enter manually."
                        )
                    # Don't show error if rate API is not available (404) - it's optional
                # If no API key, silently use manual input (no message needed)
                
                # Interest rate input - use API rate if available and selected
                if use_api_rate and api_rate:
                    interest_rate = api_rate
                    st.info(f"💰 Current Market Rate: **{api_rate*100:.2f}%** (from API Ninjas)")
                else:
                    interest_rate = st.slider("Interest Rate (%)", 2.0, 10.0, default_interest, step=0.1) / 100
                
            with inv_col2:
                st.markdown("#### 2. Income Assumptions")
                # Use your AI prediction as the baseline!
                default_nightly = st.session_state.roi_calculator_values['avg_nightly_rate']
                avg_nightly_rate = st.number_input("Avg Nightly Rate ($)", value=default_nightly, help="Defaults to AI prediction")
                default_occupancy = int(st.session_state.roi_calculator_values['occupancy_rate'] * 100)
                occupancy_rate = st.slider("Occupancy Rate (%)", 10, 90, default_occupancy) / 100
                
            with inv_col3:
                st.markdown("#### 3. Monthly Expenses")
                
                # Fetch property tax rate from API Ninjas if available
                api_tax_rate = None
                if not api_ninjas_key:
                    st.caption("ℹ️ API Ninjas key not found. Using manual property tax input.")
                elif not st.session_state.prediction_results:
                    st.caption("ℹ️ Run a price prediction first to fetch local property tax rate.")
                else:
                    # Try to use city and state from geocoding first (most accurate)
                    city = st.session_state.prediction_results.get('city')
                    state_abbr = st.session_state.prediction_results.get('state')
                    
                    # If geocoding didn't provide city/state, try to extract from search_query
                    if not city or not state_abbr:
                        search_query = st.session_state.prediction_results.get('search_query', '')
                        # Extract city and state from search query (e.g., "Los Angeles, CA" or "1155 S Grand Avenue, Los Angeles, CA")
                        if ", " in search_query:
                            parts = search_query.split(", ")
                            if len(parts) >= 2:
                                # Last part is usually state
                                potential_state = parts[-1].strip()
                                # Second to last is usually city (if address provided)
                                if len(parts) >= 3:
                                    city = parts[-2].strip()
                                else:
                                    city = parts[0].strip()
                                
                                if len(potential_state) == 2 and potential_state.isalpha():
                                    state_abbr = potential_state.upper()
                    
                    # If we have both city and state, fetch property tax
                    if city and state_abbr:
                        with st.spinner(f'Fetching property tax rate for {city}, {state_abbr}...'):
                            api_tax_rate = get_apininjas_property_tax(city, state_abbr, api_ninjas_key)
                        if not api_tax_rate:
                            st.caption(f"ℹ️ Could not fetch property tax for {city}, {state_abbr}. Check API key or city name spelling. Using manual input.")
                    else:
                        st.caption("ℹ️ Could not determine city/state from address. Using manual property tax input.")
                
                # Property tax input - use API rate if available
                default_tax = st.session_state.roi_calculator_values['property_tax_rate'] * 100
                if api_tax_rate:
                    # Convert decimal to percentage for display
                    api_tax_percent = api_tax_rate * 100
                    use_api_tax = st.checkbox(
                        f"Use local property tax rate ({api_tax_percent:.2f}%)",
                        value=True,
                        help=f"Median property tax rate for this location from API Ninjas. Uncheck to enter manually."
                    )
                    if use_api_tax:
                        property_tax_rate = api_tax_rate
                        st.caption(f"📍 Predicted Local rate: {api_tax_percent:.2f}%")
                    else:
                        property_tax_rate = st.number_input("Property Tax (%)", value=default_tax, step=0.1, help="US Avg is ~1.1%") / 100
                else:
                    property_tax_rate = st.number_input("Property Tax (%)", value=default_tax, step=0.1, help="US Avg is ~1.1%") / 100
                
                insurance_monthly = st.number_input("Insurance ($/mo)", value=int(st.session_state.roi_calculator_values['insurance_monthly']))
                utilities_monthly = st.number_input("Utilities ($/mo)", value=int(st.session_state.roi_calculator_values['utilities_monthly']))
                default_mgmt = int(st.session_state.roi_calculator_values['management_fee_pct'] * 100)
                management_fee_pct = st.slider("Management Fee (%)", 0, 25, default_mgmt, help="If you hire a property manager") / 100
            
            # Update session state automatically when values change
            st.session_state.roi_calculator_values = {
                'listing_price': listing_price,
                'down_payment_pct': down_payment_pct,
                'interest_rate': interest_rate,
                'loan_term': loan_term,
                'avg_nightly_rate': avg_nightly_rate,
                'occupancy_rate': occupancy_rate,
                'property_tax_rate': property_tax_rate,
                'insurance_monthly': insurance_monthly,
                'utilities_monthly': utilities_monthly,
                'management_fee_pct': management_fee_pct
            }

            # --- CALCULATIONS (Pure Python - No API needed) ---
            
            # Validate that listing_price is provided
            if listing_price is None:
                st.info("💡 Enter your property details above to calculate expected ROI.")
                st.stop()
            
            # A. Upfront Costs
            down_payment_cash = listing_price * down_payment_pct
            closing_costs = listing_price * 0.03 # Rule of thumb: 3% closing costs
            total_cash_invested = down_payment_cash + closing_costs
            loan_amount = listing_price - down_payment_cash
            
            # B. Monthly Mortgage (Standard Formula)
            # M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
            monthly_rate = interest_rate / 12
            num_payments = loan_term * 12
            our_mortgage_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
            
            # Optional: Validate with API Ninjas Mortgage Calculator
            api_mortgage_data = None
            api_monthly = None
            if api_ninjas_key:
                annual_property_tax = listing_price * property_tax_rate
                annual_home_insurance = insurance_monthly * 12
                api_mortgage_data = get_apininjas_mortgage_calculation(
                    loan_amount=loan_amount,
                    interest_rate=interest_rate,
                    duration_years=loan_term,
                    annual_property_tax=annual_property_tax,
                    annual_home_insurance=annual_home_insurance,
                    api_key=api_ninjas_key
                )
                if api_mortgage_data:
                    api_monthly = api_mortgage_data.get('monthly_payment', {}).get('total', 0)
            
            # Use average of our calculation and API Ninjas if available, otherwise use our calculation
            if api_monthly and api_monthly > 0:
                mortgage_payment = (our_mortgage_payment + api_monthly) / 2
            else:
                mortgage_payment = our_mortgage_payment
            
            # C. Operating Expenses
            property_tax_monthly = (listing_price * property_tax_rate) / 12
            maintenance_monthly = (listing_price * 0.01) / 12 # Rule of thumb: 1% of home value/year
            
            # D. Income
            gross_monthly_income = avg_nightly_rate * 30 * occupancy_rate
            management_fee_monthly = gross_monthly_income * management_fee_pct
            
            total_monthly_expenses = (mortgage_payment + property_tax_monthly + 
                                      insurance_monthly + utilities_monthly + 
                                      management_fee_monthly + maintenance_monthly)
            
            net_monthly_cashflow = gross_monthly_income - total_monthly_expenses
            annual_cashflow = net_monthly_cashflow * 12
            
            # E. ROI Metrics
            cash_on_cash_return = (annual_cashflow / total_cash_invested) * 100 if total_cash_invested > 0 else 0

            # --- DISPLAY RESULTS ---
            st.markdown("---")
            
            # Visual Verdict using standard "Good Deal" metrics (CoC > 8-12% is usually good)
            if cash_on_cash_return >= 12:
                st.success(f"🚀 **GREAT DEAL!** {cash_on_cash_return:.1f}% Return on Investment")
            elif cash_on_cash_return >= 8:
                st.info(f"✅ **GOOD DEAL.** {cash_on_cash_return:.1f}% Return on Investment")
            elif cash_on_cash_return > 0:
                st.warning(f"⚠️ **MARGINAL.** {cash_on_cash_return:.1f}% Return on Investment")
            else:
                st.error(f"🛑 **NEGATIVE CASH FLOW.** {cash_on_cash_return:.1f}% Return")

            # Metrics Columns
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            with res_col1:
                st.markdown("**Monthly Income**")
                st.markdown(f'<p style="color: green; font-size: 2em; font-weight: bold; margin: 0;">${gross_monthly_income:,.0f}</p>', unsafe_allow_html=True)
            with res_col2:
                st.markdown("**Total Expenses**")
                st.markdown(f'<p style="color: red; font-size: 2em; font-weight: bold; margin: 0;">${total_monthly_expenses:,.0f}</p>', unsafe_allow_html=True)
                st.caption("View monthly expenses breakdown below.")
            with res_col3:
                st.markdown("**Net Cash Flow**")
                cashflow_color = "green" if net_monthly_cashflow > 0 else "red"
                st.markdown(f'<p style="color: {cashflow_color}; font-size: 2em; font-weight: bold; margin: 0;">${net_monthly_cashflow:,.0f}</p>', unsafe_allow_html=True)
            with res_col4:
                st.markdown("**Cash to Close**")
                st.markdown(f'<p style="color: #333333; font-size: 2em; font-weight: bold; margin: 0;">${total_cash_invested:,.0f}</p>', unsafe_allow_html=True)
                st.caption("Down Payment + ~3% Closing Costs")

            # Monthly Expenses Breakdown
            with st.expander("📊 View Monthly Expenses Breakdown", expanded=True):
                # Create two columns for better layout
                breakdown_col1, breakdown_col2 = st.columns(2)
                
                # Calculate totals
                fixed_costs_total = mortgage_payment + property_tax_monthly + insurance_monthly + utilities_monthly
                variable_costs_total = maintenance_monthly + management_fee_monthly
                
                with breakdown_col1:
                    st.markdown('<p style="text-align: center; font-weight: bold; font-size: 1.1em;">Fixed Costs:</p>', unsafe_allow_html=True)
                    st.write(f"• **Mortgage Payment:** ${mortgage_payment:,.2f}")
                    st.write(f"• **Property Tax:** ${property_tax_monthly:,.2f}")
                    st.write(f"• **Home Insurance:** ${insurance_monthly:,.2f}")
                    st.write(f"• **Utilities/Wifi:** ${utilities_monthly:,.2f}")
                    st.markdown(f"**Total Fixed Costs: ${fixed_costs_total:,.2f}**")
                
                with breakdown_col2:
                    st.markdown('<p style="text-align: center; font-weight: bold; font-size: 1.1em;">Variable Costs:</p>', unsafe_allow_html=True)
                    st.write(f"• **Maintenance (1% of home value/year):** ${maintenance_monthly:,.2f}")
                    st.write(f"• **Management Fee ({management_fee_pct*100:.0f}% of income):** ${management_fee_monthly:,.2f}")
                    st.markdown(f"**Total Variable Costs: ${variable_costs_total:,.2f}**")
                
                st.markdown("---")
                st.markdown(f"**Total Monthly Expenses:** <span style='color: red; font-weight: bold;'>${total_monthly_expenses:,.2f}</span>", unsafe_allow_html=True)
            
            # Mortgage Calculator (if available)
            if api_mortgage_data and api_ninjas_key:
                with st.expander("🧮 Mortgage Calculator", expanded=False):
                    st.markdown("**Comparison with API Ninjas Mortgage Calculator:**")
                    
                    api_annual = api_mortgage_data.get('annual_payment', {}).get('total', 0)
                    api_total_interest = api_mortgage_data.get('total_interest_paid', 0)
                    
                    val_col1, val_col2, val_col3 = st.columns(3)
                    with val_col1:
                        st.markdown("**Our Calculation:**")
                        st.write(f"Monthly Payment: ${our_mortgage_payment:,.2f}")
                        st.write(f"Annual Payment: ${our_mortgage_payment * 12:,.2f}")
                    
                    with val_col2:
                        st.markdown("**API Ninjas:**")
                        st.write(f"Monthly Payment: ${api_monthly:,.2f}")
                        st.write(f"Annual Payment: ${api_annual:,.2f}")
                        if api_total_interest:
                            st.write(f"Total Interest: ${api_total_interest:,.2f}")
                    
                    with val_col3:
                        st.markdown("**Average Used:**")
                        st.write(f"Monthly Payment: ${mortgage_payment:,.2f}")
                        st.write(f"Annual Payment: ${mortgage_payment * 12:,.2f}")
                    
                    # Show difference if there is one
                    if abs(our_mortgage_payment - api_monthly) > 1:
                        diff = our_mortgage_payment - api_monthly
                        st.caption(f"💡 Difference: ${diff:,.2f}/month between calculations. Using average for fixed costs.")
                    else:
                        st.success("✅ Calculations match!")
            
            # Rental Comparables Section (Top 5)
            # Get rental comparables from session state if available
            rental_comparables_data = st.session_state.get('rental_comparables')
            
            # Debug: Always show what's happening with comparables
            # Check if rental comparables are available (debug messages removed)
            
            if rental_comparables_data:
                st.markdown("---")
                st.subheader("📊 Comparable Rental Properties")
                
                comparables = rental_comparables_data.get('comparables', [])
                if comparables and len(comparables) > 0:
                    # Filter out comparables with invalid rent (0 or less than $100)
                    def get_rent_value(comp):
                        """Extract rent value trying multiple field names"""
                        # Try all possible field names for rent - check for None explicitly, not just falsy
                        rent = None
                        for field in ['price', 'rent', 'monthlyRent', 'rentAmount', 'estimatedRent', 'listedRent', 'rentPrice', 'monthlyPrice', 'rentalPrice', 'rentValue']:
                            value = comp.get(field)
                            if value is not None and value != 0:  # Explicitly check for None and 0
                                rent = value
                                break
                        
                        # Try nested structures if still not found
                        if rent is None:
                            if isinstance(comp.get('rental'), dict):
                                rent = comp.get('rental').get('rent')
                            elif isinstance(comp.get('pricing'), dict):
                                rent = comp.get('pricing').get('rent')
                        
                        # Convert to float if it's a string
                        if isinstance(rent, str):
                            try:
                                # Remove $ and commas
                                rent = float(rent.replace('$', '').replace(',', '').strip())
                            except:
                                rent = 0
                        
                        # Return 0 if still None, otherwise return as float
                        return float(rent) if rent is not None else 0
                    
                    # Filter comparables: rent must be > 100
                    valid_comparables = [
                        comp for comp in comparables 
                        if get_rent_value(comp) > 100
                    ]
                    
                    # Show how many were filtered
                    filtered_count = len(comparables) - len(valid_comparables)
                    if filtered_count > 0:
                        st.caption(f"ℹ️ Filtered out {filtered_count} comparables with rent ≤ $100/month")
                    
                    if not valid_comparables:
                        st.info("ℹ️ No comparable properties found with valid rent data (>$100/month).")
                    else:
                        # Get top 10 comparables (sorted by rent, lowest first) - don't filter by status
                        top_10 = sorted(valid_comparables, key=lambda x: get_rent_value(x), reverse=False)[:10]
                    
                    if valid_comparables and top_10:
                        # Create DataFrame for visualization with enhanced RentCast data
                        # RentCast API field names may vary - try multiple possible field names
                        comp_df = pd.DataFrame([
                            {
                                'Address': (
                                    comp.get('formattedAddress') or 
                                    comp.get('address') or 
                                    comp.get('streetAddress') or 
                                    f"{comp.get('addressLine1', '')} {comp.get('addressLine2', '')}".strip() or
                                    f"{comp.get('streetNumber', '')} {comp.get('streetName', '')}".strip() or
                                    'N/A'
                                ),
                                'Listed Rent': get_rent_value(comp),
                                'Bedrooms': comp.get('bedrooms') or comp.get('bedroomCount') or comp.get('beds') or 'N/A',
                                'Bathrooms': comp.get('bathrooms') or comp.get('bathroomCount') or comp.get('baths') or 'N/A',
                                'Square Feet': comp.get('squareFootage') or comp.get('squareFeet') or comp.get('sqft') or comp.get('livingArea') or 'N/A',
                                'Distance (mi)': round(comp.get('distance', 0), 2) if comp.get('distance') is not None else 'N/A',
                                # Additional RentCast fields
                                'Price/sqft': round(get_rent_value(comp) / (comp.get('squareFootage') or comp.get('squareFeet') or comp.get('sqft') or comp.get('livingArea') or 1), 2) if (comp.get('squareFootage') or comp.get('squareFeet') or comp.get('sqft') or comp.get('livingArea')) else 'N/A',
                                'Similarity (%)': round(comp.get('correlation', 0) * 100, 1) if comp.get('correlation') is not None else (comp.get('similarity', 0) * 100 if comp.get('similarity') is not None else comp.get('similarityScore', 'N/A')),
                                'Last Seen': comp.get('lastSeen') or comp.get('lastSeenDate') or comp.get('dateSeen') or 'N/A',
                                'Days Ago': comp.get('daysAgo') or comp.get('daysSinceSeen') or 'N/A',
                                'Property Type': comp.get('propertyType') or comp.get('type') or comp.get('propertyTypeName') or 'N/A',
                                'Year Built': comp.get('yearBuilt') or comp.get('builtYear') or 'N/A'
                            }
                            for comp in top_10
                        ])
                        
                        # Display rent estimate for the property
                        # Try multiple possible field names for rent estimate
                        property_rent_estimate = (
                            rental_comparables_data.get('rent') or 
                            rental_comparables_data.get('monthlyRent') or 
                            rental_comparables_data.get('rentAmount') or 
                            rental_comparables_data.get('estimatedRent') or
                            0
                        )
                        if property_rent_estimate:
                            # Get price prediction from session state for comparison
                            ai_prediction = st.session_state.prediction_results.get('price_prediction', 0) if st.session_state.prediction_results else 0
                            st.info(f"💰 **Estimated Monthly Rent (Long-Term): ${property_rent_estimate:,.0f}**")
                            if ai_prediction > 0:
                                st.caption(f"ℹ️ **Note:** This is for traditional long-term rental (12+ month lease). Your AI prediction above (${ai_prediction:,.0f}/night) is for short-term Airbnb rental, which typically generates higher monthly income due to premium pricing and flexibility.")
                            else:
                                st.caption("ℹ️ **Note:** This is for traditional long-term rental (12+ month lease), different from short-term Airbnb rental pricing.")
                        
                        # Create bar chart for rental prices
                        # Use 'Listed Rent' instead of 'Monthly Rent'
                        chart_df = comp_df.copy()
                        chart_df['Rent'] = chart_df['Listed Rent']  # For chart compatibility
                        
                        chart = alt.Chart(chart_df).mark_bar(color='#1f77b4').encode(
                            x=alt.X('Rent:Q', title='Monthly Rent ($)', axis=alt.Axis(format='$,.0f')),
                            y=alt.Y('Address:N', title='Property Address', sort='-x'),
                            tooltip=['Address', 'Rent:Q', 'Bedrooms', 'Bathrooms', 'Square Feet', 'Price/sqft', 'Similarity (%)', 'Distance (mi)']
                        ).properties(
                            width=700,
                            height=300,
                            title='Top 5 Comparable Rental Properties - Monthly Rent'
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Display detailed table with enhanced RentCast format
                        with st.expander("📋 View Detailed Comparable Properties", expanded=True):
                            # Format the DataFrame for better display (matching RentCast format)
                            display_df = comp_df.copy()
                            
                            # Format rent with price per sqft
                            def format_rent_with_sqft(row):
                                rent = row['Listed Rent']
                                sqft = row['Square Feet']
                                price_per_sqft = row['Price/sqft']
                                
                                if isinstance(rent, (int, float)) and isinstance(price_per_sqft, (int, float)):
                                    return f"${rent:,.0f} (${price_per_sqft:.2f} /ft²)"
                                elif isinstance(rent, (int, float)):
                                    return f"${rent:,.0f}"
                                else:
                                    return str(rent)
                            
                            display_df['Listed Rent'] = display_df.apply(format_rent_with_sqft, axis=1)
                            
                            # Format similarity with % if it's a number
                            if 'Similarity (%)' in display_df.columns:
                                display_df['Similarity (%)'] = display_df['Similarity (%)'].apply(
                                    lambda x: f"{x}%" if isinstance(x, (int, float)) else str(x)
                                )
                            
                            # Format last seen with days ago
                            def format_last_seen(row):
                                last_seen = row['Last Seen']
                                days_ago = row['Days Ago']
                                
                                if last_seen != 'N/A' and days_ago != 'N/A':
                                    return f"{last_seen} ({days_ago} Days Ago)"
                                elif last_seen != 'N/A':
                                    return str(last_seen)
                                elif days_ago != 'N/A':
                                    return f"{days_ago} Days Ago"
                                else:
                                    return 'N/A'
                            
                            display_df['Last Seen'] = display_df.apply(format_last_seen, axis=1)
                            
                            # Format property type with year built
                            def format_property_type(row):
                                prop_type = row['Property Type']
                                year = row['Year Built']
                                
                                if prop_type != 'N/A' and year != 'N/A':
                                    return f"{prop_type}, Built {year}"
                                elif prop_type != 'N/A':
                                    return str(prop_type)
                                elif year != 'N/A':
                                    return f"Built {year}"
                                else:
                                    return 'N/A'
                            
                            display_df['Type'] = display_df.apply(format_property_type, axis=1)
                            
                            # Select and reorder columns for display (matching RentCast format)
                            display_columns = ['Address', 'Listed Rent', 'Last Seen', 'Similarity (%)', 'Distance (mi)', 
                                             'Bedrooms', 'Bathrooms', 'Square Feet', 'Type']
                            
                            # Only include columns that exist
                            available_columns = [col for col in display_columns if col in display_df.columns]
                            display_df = display_df[available_columns]
                            
                            # Rename for better display
                            display_df = display_df.rename(columns={
                                'Listed Rent': 'Listed Rent',
                                'Similarity (%)': 'Similarity',
                                'Distance (mi)': 'Distance',
                                'Square Feet': 'Sq.Ft.',
                                'Bedrooms': 'Beds',
                                'Bathrooms': 'Baths'
                            })
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No comparable properties found in RentCast database.")
                else:
                    st.info("No comparable rental properties available for this address.")