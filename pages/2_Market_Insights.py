import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import time
from sidebar import show_sidebar

# Set page config
st.set_page_config(
    page_title="📈 Market Insights - Pricision AI",
    page_icon="images/logo.png",
    layout="wide"
)

# Show sidebar
show_sidebar()

# --- TABLE CONFIGURATION ---
# Configure which BigQuery table to use for different sections
TABLE_CONFIG = {
    'all_quarters': 'airbnb-dash-479208.US_all.streamlit_dashboard_master',
    'quarter_4': 'airbnb-dash-479208.US_all.streamlit_master_dashboard_quarter_4'
}

# --- BIGQUERY AUTHENTICATION ---
@st.cache_resource
def get_bigquery_client():
    """
    Initialize BigQuery client using service account credentials.
    Reuses the same authentication pattern from main app.
    """
    try:
        # Option 1: Try Streamlit Secrets - Official pattern using 'gcp_service_account'
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

# --- QUERY TEMPORAL DATA (ENSURES ALL QUARTERS) ---
@st.cache_data(ttl=3600)
def query_temporal_data(filters_dict, table_name, sample_percent=10):
    """
    Query BigQuery for temporal trends data, ensuring all quarters are represented.
    Uses UNION ALL to sample from each quarter separately.
    """
    client = get_bigquery_client()
    
    if client is None:
        return pd.DataFrame()
    
    try:
        # Build WHERE clause dynamically (excluding quarter filter)
        where_conditions = []
        
        # Location filter
        if filters_dict.get('locations') and len(filters_dict['locations']) > 0:
            locations_str = "', '".join(filters_dict['locations'])
            where_conditions.append(f"src_location IN ('{locations_str}')")
        
        # Property type filter
        if filters_dict.get('property_types') and len(filters_dict['property_types']) > 0:
            prop_types_str = "', '".join(filters_dict['property_types'])
            where_conditions.append(f"property_category IN ('{prop_types_str}')")
        
        # Price range filter
        if filters_dict.get('min_price') is not None:
            where_conditions.append(f"price >= {filters_dict['min_price']}")
        if filters_dict.get('max_price') is not None:
            where_conditions.append(f"price <= {filters_dict['max_price']}")
        
        # Host status filter
        if filters_dict.get('superhost') != 'All':
            is_superhost = 1 if filters_dict['superhost'] == 'Yes' else 0
            where_conditions.append(f"host_is_superhost = {is_superhost}")
        
        # Combine WHERE conditions
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # For temporal trends, we need to ensure all quarters are represented
        # Query each quarter separately to guarantee equal representation
        # Calculate rows per quarter (aim for ~12.5K per quarter for 50K total)
        rows_per_quarter = min(12500, max(2500, int(900000 * (sample_percent / 100) / 4)))
        
        # Build the query parts - use clean single-line format for each SELECT
        select_fields = "host_response_time, host_is_superhost, src_location, property_category, accommodates, bathrooms, bedrooms, accommodates_per_bedroom, price, number_of_reviews_l30d, review_scores_rating, review_scores_cleanliness, review_scores_location, review_scores_value, reviews_per_month, instant_bookable, CAST(quarter AS INT64) AS quarter"
        
        query_parts = []
        for quarter in [1, 2, 3, 4]:
            # Build WHERE clause for this quarter - combine all conditions properly
            additional_conditions = [
                "price IS NOT NULL",
                "price > 0",
                "property_category != 'Hotel/Resort'",
                f"CAST(quarter AS INT64) = {quarter}",
                f"MOD(ABS(FARM_FINGERPRINT(CAST(price AS STRING) || CAST(accommodates AS STRING) || COALESCE(src_location, ''))), 100) < {sample_percent}"
            ]
            
            if where_clause and where_clause != "1=1":
                all_conditions = [where_clause] + additional_conditions
            else:
                all_conditions = additional_conditions
            
            quarter_where = " AND ".join(all_conditions)
            
            # Build clean query part
            query_part = f"SELECT {select_fields} FROM `{table_name}` WHERE {quarter_where} LIMIT {rows_per_quarter}"
            query_parts.append(query_part)
        
        # Combine all quarter queries with UNION ALL
        query = " UNION ALL ".join([f"({q})" for q in query_parts])
        
        # Execute query with timeout handling
        query_job = client.query(query, job_config=bigquery.QueryJobConfig(
            maximum_bytes_billed=10**10,  # 10GB limit
            use_query_cache=True,
            use_legacy_sql=False
        ))
        
        # Wait for results with timeout
        try:
            query_start_time = time.time()
            timeout_seconds = 120  # 2 minute timeout
            
            while not query_job.done():
                if time.time() - query_start_time > timeout_seconds:
                    query_job.cancel()
                    raise TimeoutError(f"Query exceeded {timeout_seconds} second timeout")
                time.sleep(1)
            
            if query_job.errors:
                raise Exception(f"Query error: {query_job.errors}")
            
            results = query_job.result()
        except TimeoutError as e:
            st.error(f"Query timeout: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in results])
        
        return df
        
    except Exception as e:
        st.error(f"Error querying temporal data: {str(e)}")
        return pd.DataFrame()

# --- QUERY INSIGHTS DATA ---
@st.cache_data(ttl=3600)
def query_insights_data(filters_dict, table_name, sample_percent=10, additional_where=None):
    """
    Query BigQuery for insights data with dynamic filters.
    
    Args:
        filters_dict: Dictionary of filter values
        table_name: Name of the BigQuery table to query
        sample_percent: Percentage of data to sample (default 10% for performance)
        additional_where: Additional WHERE conditions as string
    
    Returns:
        DataFrame with filtered data
    """
    client = get_bigquery_client()
    
    if client is None:
        return pd.DataFrame()
    
    try:
        # Build WHERE clause dynamically
        where_conditions = []
        
        # Location filter
        if filters_dict.get('locations') and len(filters_dict['locations']) > 0:
            locations_str = "', '".join(filters_dict['locations'])
            where_conditions.append(f"src_location IN ('{locations_str}')")
        
        # Property type filter
        if filters_dict.get('property_types') and len(filters_dict['property_types']) > 0:
            prop_types_str = "', '".join(filters_dict['property_types'])
            where_conditions.append(f"property_category IN ('{prop_types_str}')")
        
        # Price range filter
        if filters_dict.get('min_price') is not None:
            where_conditions.append(f"price >= {filters_dict['min_price']}")
        if filters_dict.get('max_price') is not None:
            where_conditions.append(f"price <= {filters_dict['max_price']}")
        
        # Host status filter
        if filters_dict.get('superhost') != 'All':
            is_superhost = 1 if filters_dict['superhost'] == 'Yes' else 0
            where_conditions.append(f"host_is_superhost = {is_superhost}")
        
        # Date/Quarter filter - only apply if quarters list is not empty
        quarters_list = filters_dict.get('quarters', [])
        if quarters_list and len(quarters_list) > 0:
            # Convert to integers to handle string/numeric issues
            quarters_int = [int(q) for q in quarters_list if q is not None]
            if quarters_int:
                quarters_str = ", ".join([str(q) for q in quarters_int])
                where_conditions.append(f"CAST(quarter AS INT64) IN ({quarters_str})")
        
        # Add additional WHERE conditions if provided
        if additional_where:
            where_conditions.append(additional_where)
        
        # Combine WHERE conditions
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Build query with sampling for performance
        # Only select columns actually used in visualizations to reduce data transfer
        sample_clause = f"TABLESAMPLE SYSTEM ({sample_percent} PERCENT)" if sample_percent < 100 else ""
        # Add LIMIT as safety measure - even with sampling, limit to prevent timeouts
        # Calculate reasonable limit based on sample size (max 50K rows)
        max_rows = min(50000, max(10000, int(900000 * (sample_percent / 100))))
        limit_clause = f"LIMIT {max_rows}"
        
        query = f"""
        SELECT 
            -- Host metrics
            host_response_time,
            host_is_superhost,
            -- Location
            src_location,
            -- Property characteristics
            property_category,
            accommodates,
            bathrooms,
            bedrooms,
            accommodates_per_bedroom,
            -- Pricing
            price,
            -- Reviews
            number_of_reviews_l30d,
            review_scores_rating,
            review_scores_cleanliness,
            review_scores_location,
            review_scores_value,
            reviews_per_month,
            -- Booking
            instant_bookable,
            -- Temporal
            CAST(quarter AS INT64) AS quarter
        FROM 
            `{table_name}`
        {sample_clause}
        WHERE 
            {where_clause}
            AND price IS NOT NULL
            AND price > 0
            AND property_category != 'Hotel/Resort'
        {limit_clause}
        """
        
        # Execute query with timeout handling
        query_job = client.query(query, job_config=bigquery.QueryJobConfig(
            maximum_bytes_billed=10**10,  # 10GB limit
            use_query_cache=True,
            use_legacy_sql=False
        ))
        
        # Wait for results with shorter timeout and better error handling
        try:
            # Use a shorter timeout and check job status
            query_start_time = time.time()
            timeout_seconds = 120  # 2 minute timeout
            
            while not query_job.done():
                if time.time() - query_start_time > timeout_seconds:
                    query_job.cancel()
                    raise TimeoutError(f"Query exceeded {timeout_seconds} second timeout")
                time.sleep(1)  # Check every second
            
            if query_job.errors:
                raise Exception(f"Query error: {query_job.errors}")
            
            results = query_job.result()
        except TimeoutError as e:
            st.error(f"Query timeout: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in results])
        
        return df
        
    except Exception as e:
        st.error(f"Error querying BigQuery: {str(e)}")
        return pd.DataFrame()

# --- GET UNIQUE VALUES FOR FILTERS ---
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_filter_options():
    """Get unique values for filter dropdowns."""
    client = get_bigquery_client()
    
    if client is None:
        return {
            'locations': [],
            'property_types': [],
            'quarters': []
        }
    
    try:
        # Use the all_quarters table for filter options
        table_name = TABLE_CONFIG['all_quarters']
        
        # Get unique locations
        locations_query = f"""
        SELECT DISTINCT src_location
        FROM `{table_name}`
        WHERE src_location IS NOT NULL
        ORDER BY src_location
        LIMIT 100
        """
        
        # Get unique property types (excluding Hotel/Resort)
        property_types_query = f"""
        SELECT DISTINCT property_category
        FROM `{table_name}`
        WHERE property_category IS NOT NULL
            AND property_category != 'Hotel/Resort'
        ORDER BY property_category
        """
        
        # Get unique quarters
        quarters_query = f"""
        SELECT DISTINCT quarter
        FROM `{table_name}`
        WHERE quarter IS NOT NULL
        ORDER BY quarter
        """
        
        locations_df = pd.DataFrame([dict(row) for row in client.query(locations_query).result()])
        property_types_df = pd.DataFrame([dict(row) for row in client.query(property_types_query).result()])
        quarters_df = pd.DataFrame([dict(row) for row in client.query(quarters_query).result()])
        
        return {
            'locations': locations_df['src_location'].tolist() if not locations_df.empty else [],
            'property_types': property_types_df['property_category'].tolist() if not property_types_df.empty else [],
            'quarters': sorted(quarters_df['quarter'].tolist()) if not quarters_df.empty else []
        }
        
    except Exception as e:
        st.warning(f"Error loading filter options: {str(e)}")
        return {
            'locations': [],
            'property_types': [],
            'quarters': []
        }

# --- MAIN PAGE LAYOUT ---
st.title("📈 Market Insights Dashboard")
st.markdown("Explore insights from over 1 million Airbnb listings across the US")

# Initialize session state for filters
if 'insights_filters' not in st.session_state:
    st.session_state.insights_filters = {
        'locations': [],  # Empty = all locations
        'property_types': [],  # Empty = all property types
        'quarters': [1, 2, 3, 4],  # Always all quarters
        'superhost': 'All'  # Always available, default to 'All' (no filtering)
    }

# Initialize sample_size in session state if not exists
if 'sample_size' not in st.session_state:
    st.session_state.sample_size = 10  # Default to 10%

# Initialize selected_locations in session state
if 'selected_locations' not in st.session_state:
    st.session_state.selected_locations = []

# --- FILTERS SECTION (Top of page, collapsed by default) ---
with st.expander("🔍 Market Insights Filters", expanded=False):
    # Get filter options lazily - only when filters section is expanded
    with st.spinner("Loading filter options..."):
        filter_options = get_filter_options()
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Location filter - default to all locations (empty list)
        selected_locations = st.multiselect(
            "Location",
            options=filter_options['locations'],
            default=st.session_state.selected_locations,  # Use session state
            key="filter_locations"
        )
        st.session_state.selected_locations = selected_locations  # Update session state
    
    with filter_col2:
        # Sample size selector - default to 10
        sample_size = st.slider(
            "Data Sample Size (%)",
            min_value=1,
            max_value=100,
            value=st.session_state.sample_size,  # Use session state value
            step=1,
            help="Lower percentage = faster queries. Higher = more accurate. Recommended: 1-5% for quick results, 10%+ may timeout.",
            key="filter_sample_size"
        )
        st.session_state.sample_size = sample_size  # Update session state
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

# Get sample_size from session state (defaults to 10 if not set)
sample_size = st.session_state.get('sample_size', 10)

# Get selected_locations from session state
selected_locations = st.session_state.get('selected_locations', [])

# Update session state (always show all property types and all quarters)
st.session_state.insights_filters = {
    'locations': selected_locations,  # Empty = all locations
    'property_types': [],  # Always all property types
    'quarters': [1, 2, 3, 4],  # Always all quarters
    'superhost': 'All'  # Always set to 'All' so superhost data is always available
}

st.markdown("---")

# --- PRICING INSIGHTS SECTION ---
try:
    # Default to all quarters for pricing insights
    pricing_quarters = [1, 2, 3, 4]
    
    # Query pricing data - always use all_quarters table for all quarters
    pricing_table = TABLE_CONFIG['all_quarters']
    pricing_filters = st.session_state.insights_filters.copy()
    pricing_filters['quarters'] = pricing_quarters
    
    with st.spinner("Loading pricing data..."):
        pricing_df = query_insights_data(pricing_filters, pricing_table, sample_size)
    
    if pricing_df.empty:
        st.warning("No pricing data found for selected quarters.")
    else:
        # Price Statistics - Display above Pricing Insights
        st.markdown("#### Price Statistics")
        price_stats_col1, price_stats_col2, price_stats_col3, price_stats_col4 = st.columns(4)
        
        with price_stats_col1:
            mean_val = round(pricing_df['price'].mean(), 3)
            st.metric("Mean", f"${mean_val:,.3f}")
        
        with price_stats_col2:
            median_val = round(pricing_df['price'].median(), 3)
            st.metric("Median", f"${median_val:,.3f}")
        
        with price_stats_col3:
            q25_val = round(pricing_df['price'].quantile(0.25), 3)
            st.metric("25th Percentile", f"${q25_val:,.3f}")
        
        with price_stats_col4:
            q75_val = round(pricing_df['price'].quantile(0.75), 3)
            st.metric("75th Percentile", f"${q75_val:,.3f}")
        
        st.markdown("---")  # Separator line
        
        with st.expander("💰 Pricing Insights", expanded=True):
            col1, col2 = st.columns(2)
        
            with col1:
                # Price Distribution Histogram
                price_hist = alt.Chart(pricing_df).mark_bar().encode(
                    alt.X("price:Q", bin=alt.Bin(maxbins=50), title="Price ($)"),
                    alt.Y("count()", title="Number of Listings"),
                    color=alt.value("#FF4444")
                ).properties(
                    title="Price Distribution",
                    width=400,
                    height=300
                )
                st.altair_chart(price_hist, use_container_width=True)
            
            with col2:
                # Price by Property Type
                if 'property_category' in pricing_df.columns and not pricing_df['property_category'].isna().all():
                    price_by_type = alt.Chart(pricing_df).mark_boxplot().encode(
                        alt.X("property_category:N", title="Property Type", axis=alt.Axis(labelAngle=0)),
                        alt.Y("price:Q", title="Price ($)"),
                        color=alt.value("#FF4444")
                    ).properties(
                        title="Price by Property Type",
                        width=400,
                        height=300
                    )
                    st.altair_chart(price_by_type, use_container_width=True)
                else:
                    st.info("Property type data not available")
            
            # Price by Location (Top 20) - from ALL quarters
            with st.spinner("Loading location data from all quarters..."):
                location_filters = st.session_state.insights_filters.copy()
                location_filters['quarters'] = []  # All quarters
                location_df = query_insights_data(location_filters, TABLE_CONFIG['all_quarters'], sample_size)
            
            if 'src_location' in location_df.columns and not location_df['src_location'].isna().all():
                location_avg_price = location_df.groupby('src_location')['price'].mean().sort_values(ascending=False).head(20).reset_index()
                # Format to 2 decimal places
                location_avg_price['price_label'] = location_avg_price['price'].apply(lambda x: f"${x:.2f}")
                # Calculate middle position for centering text (half of price value)
                location_avg_price['price_mid'] = location_avg_price['price'] / 2
                
                # Create bar chart
                bars = alt.Chart(location_avg_price).mark_bar().encode(
                    alt.X("price:Q", title="Average Price ($)"),
                    alt.Y("src_location:N", sort="-x", title="Location"),
                    color=alt.value("#FF4444"),
                    tooltip=["src_location:N", alt.Tooltip("price:Q", format=".2f", title="Average Price ($)")]
                )
                
                # Add labels inside bars (centered)
                labels = alt.Chart(location_avg_price).mark_text(
                    align='center',
                    baseline='middle',
                    fontSize=10,
                    fontWeight='bold',
                    color='white'
                ).encode(
                    alt.X("price_mid:Q", title="Average Price ($)"),
                    alt.Y("src_location:N", sort="-x", title="Location"),
                    text="price_label:N"
                )
                
                price_by_location = (bars + labels).properties(
                    title="Average Price by Location (Top 20) - All Quarters",
                    width=800,
                    height=500
                )
                st.altair_chart(price_by_location, use_container_width=True)
            
            # Price vs Accommodates per Bedroom (combined chart)
            if 'accommodates_per_bedroom' in pricing_df.columns and 'price' in pricing_df.columns:
                price_vs_accommodates_per_bedroom = alt.Chart(pricing_df.sample(min(1000, len(pricing_df)))).mark_circle(size=50, opacity=0.5).encode(
                    alt.X("accommodates_per_bedroom:Q", title="Accommodates per Bedroom"),
                    alt.Y("price:Q", title="Price ($)"),
                    color=alt.value("#FF4444")
                ).properties(
                    title="Price vs Accommodates per Bedroom",
                    width=800,
                    height=300
                )
                st.altair_chart(price_vs_accommodates_per_bedroom, use_container_width=True)
except Exception as e:
    st.error(f"Error displaying pricing insights: {str(e)}")

# --- HOST PERFORMANCE SECTION ---
try:
    with st.expander("👤 Host Performance", expanded=True):
        # Host Performance uses all quarters
        with st.spinner("Loading host performance data from all quarters..."):
            host_filters = st.session_state.insights_filters.copy()
            host_filters['quarters'] = []  # All quarters
            host_df = query_insights_data(host_filters, TABLE_CONFIG['all_quarters'], sample_size)
        
        if host_df.empty:
            st.warning("No host performance data found.")
        else:
            col1, col2 = st.columns(2)
        
            with col1:
                # Response Time Distribution
                if 'host_response_time' in host_df.columns and not host_df['host_response_time'].isna().all():
                    response_time_counts = host_df['host_response_time'].value_counts().reset_index()
                    response_time_counts.columns = ['response_time', 'count']
                    response_time_chart = alt.Chart(response_time_counts).mark_bar().encode(
                        alt.X("response_time:N", title="Response Time", axis=alt.Axis(labelAngle=0)),
                        alt.Y("count:Q", title="Number of Hosts"),
                        color=alt.value("#4CAF50")
                    ).properties(
                        title="Host Response Time Distribution",
                        width=400,
                        height=300
                    )
                    st.altair_chart(response_time_chart, use_container_width=True)
                else:
                    st.info("Response time data not available")
            
            with col2:
                # Superhost Impact (moved here since response rate is removed)
                if 'host_is_superhost' in host_df.columns:
                    superhost_comparison = host_df.groupby('host_is_superhost').agg({
                        'price': 'mean',
                        'review_scores_rating': 'mean'
                    }).reset_index()
                    superhost_comparison['host_is_superhost'] = superhost_comparison['host_is_superhost'].map({0: 'Not Superhost', 1: 'Superhost'})
                    
                    st.markdown("#### Superhost Impact")
                    superhost_col1, superhost_col2 = st.columns(2)
                    with superhost_col1:
                        st.metric("Avg Price (Superhost)", f"${superhost_comparison[superhost_comparison['host_is_superhost']=='Superhost']['price'].values[0]:.3f}" if len(superhost_comparison[superhost_comparison['host_is_superhost']=='Superhost']) > 0 else "N/A")
                        st.metric("Avg Price (Not Superhost)", f"${superhost_comparison[superhost_comparison['host_is_superhost']=='Not Superhost']['price'].values[0]:.3f}" if len(superhost_comparison[superhost_comparison['host_is_superhost']=='Not Superhost']) > 0 else "N/A")
                    with superhost_col2:
                        st.metric("Avg Rating (Superhost)", f"{superhost_comparison[superhost_comparison['host_is_superhost']=='Superhost']['review_scores_rating'].values[0]:.3f}" if len(superhost_comparison[superhost_comparison['host_is_superhost']=='Superhost']) > 0 else "N/A")
                        st.metric("Avg Rating (Not Superhost)", f"{superhost_comparison[superhost_comparison['host_is_superhost']=='Not Superhost']['review_scores_rating'].values[0]:.3f}" if len(superhost_comparison[superhost_comparison['host_is_superhost']=='Not Superhost']) > 0 else "N/A")
except Exception as e:
    st.error(f"Error displaying host performance insights: {str(e)}")

# --- REVIEW ANALYSIS SECTION ---
try:
    with st.expander("⭐ Review Analysis", expanded=True):
        # Review Analysis uses all quarters - use temporal data query to ensure all quarters
        with st.spinner("Loading review data from all quarters..."):
            review_filters = st.session_state.insights_filters.copy()
            review_filters['quarters'] = []  # All quarters
            review_df = query_temporal_data(review_filters, TABLE_CONFIG['all_quarters'], sample_size)
        
        if review_df.empty:
            st.warning("No review data found.")
        else:
            # Ensure quarter is numeric for proper sorting
            if 'quarter' in review_df.columns:
                review_df['quarter'] = pd.to_numeric(review_df['quarter'], errors='coerce')
            
            col1, col2 = st.columns(2)
        
            with col1:
                # Overall Rating Distribution
                if 'review_scores_rating' in review_df.columns and not review_df['review_scores_rating'].isna().all():
                    rating_hist = alt.Chart(review_df).mark_bar().encode(
                        alt.X("review_scores_rating:Q", bin=alt.Bin(maxbins=20), title="Rating"),
                        alt.Y("count()", title="Number of Listings"),
                        color=alt.value("#FFA500")
                    ).properties(
                        title="Overall Rating Distribution",
                        width=400,
                        height=300
                    )
                    st.altair_chart(rating_hist, use_container_width=True)
            
            with col2:
                # Review Score Breakdown
                review_cols = ['review_scores_cleanliness', 'review_scores_location', 'review_scores_value']
                review_data = []
                for col in review_cols:
                    if col in review_df.columns and not review_df[col].isna().all():
                        review_data.append({
                            'metric': col.replace('review_scores_', '').title(),
                            'average': round(review_df[col].mean(), 3)
                        })
                
                if review_data:
                    review_data_df = pd.DataFrame(review_data)
                    # Format to 3 decimal places
                    review_data_df['average_label'] = review_data_df['average'].apply(lambda x: f"{x:.3f}")
                    # Calculate middle position for centering text
                    review_data_df['average_mid'] = review_data_df['average'] / 2
                    
                    # Create bar chart
                    bars = alt.Chart(review_data_df).mark_bar().encode(
                        alt.X("metric:N", title="Review Category", axis=alt.Axis(labelAngle=0)),
                        alt.Y("average:Q", title="Average Score"),
                        color=alt.value("#FFA500"),
                        tooltip=["metric:N", alt.Tooltip("average:Q", format=".3f", title="Average Score")]
                    )
                    
                    # Add labels inside bars (centered)
                    labels = alt.Chart(review_data_df).mark_text(
                        align='center',
                        baseline='middle',
                        fontSize=11,
                        fontWeight='bold',
                        color='white'
                    ).encode(
                        alt.X("metric:N", title="Review Category", axis=alt.Axis(labelAngle=0)),
                        alt.Y("average_mid:Q", title="Average Score"),
                        text="average_label:N"
                    )
                    
                    review_chart = (bars + labels).properties(
                        title="Average Review Scores by Category",
                        width=400,
                        height=300
                    )
                    st.altair_chart(review_chart, use_container_width=True)
            
            # Review Volume Trends by Quarter (Q1-4) - Line chart with point labels
            if 'quarter' in review_df.columns and 'reviews_per_month' in review_df.columns:
                reviews_by_quarter = review_df.groupby('quarter')['reviews_per_month'].mean().reset_index()
                reviews_by_quarter = reviews_by_quarter.sort_values('quarter')
                # Format reviews for display
                reviews_by_quarter['reviews_label'] = reviews_by_quarter['reviews_per_month'].apply(lambda x: f"{x:.3f}")
                
                # Create line chart
                line = alt.Chart(reviews_by_quarter).mark_line(point=True, strokeWidth=3).encode(
                    alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                    alt.Y("reviews_per_month:Q", title="Average Reviews per Month"),
                    color=alt.value("#FFA500"),
                    tooltip=["quarter:O", alt.Tooltip("reviews_per_month:Q", format=".3f", title="Average Reviews per Month")]
                )
                
                # Add text labels at each point
                labels = alt.Chart(reviews_by_quarter).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-10,
                    fontSize=12,
                    fontWeight='bold'
                ).encode(
                    alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                    alt.Y("reviews_per_month:Q", title="Average Reviews per Month"),
                    text="reviews_label:N",
                    color=alt.value("#FFA500")
                )
                
                reviews_trend = (line + labels).properties(
                    title="Review Volume Trends by Quarter (Reviews per Month)",
                    width=800,
                    height=300
                )
                st.altair_chart(reviews_trend, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                # Recent Reviews (L30D) - all data
                if 'number_of_reviews_l30d' in review_df.columns and 'property_category' in review_df.columns:
                    recent_reviews = review_df.groupby('property_category')['number_of_reviews_l30d'].mean().reset_index()
                    # Format to 3 decimal places
                    recent_reviews['reviews_label'] = recent_reviews['number_of_reviews_l30d'].apply(lambda x: f"{x:.3f}")
                    # Calculate middle position for centering text
                    recent_reviews['reviews_mid'] = recent_reviews['number_of_reviews_l30d'] / 2
                    
                    # Create bar chart
                    bars = alt.Chart(recent_reviews).mark_bar().encode(
                        alt.X("property_category:N", title="Property Type", axis=alt.Axis(labelAngle=0)),
                        alt.Y("number_of_reviews_l30d:Q", title="Avg Reviews (Last 30 Days)"),
                        color=alt.value("#FFA500"),
                        tooltip=["property_category:N", alt.Tooltip("number_of_reviews_l30d:Q", format=".3f", title="Avg Reviews (Last 30 Days)")]
                    )
                    
                    # Add labels inside bars (centered)
                    labels = alt.Chart(recent_reviews).mark_text(
                        align='center',
                        baseline='middle',
                        fontSize=11,
                        fontWeight='bold',
                        color='white'
                    ).encode(
                        alt.X("property_category:N", title="Property Type", axis=alt.Axis(labelAngle=0)),
                        alt.Y("reviews_mid:Q", title="Avg Reviews (Last 30 Days)"),
                        text="reviews_label:N"
                    )
                    
                    recent_reviews_chart = (bars + labels).properties(
                        title="Recent Reviews by Property Type (All Data)",
                        width=400,
                        height=300
                    )
                    st.altair_chart(recent_reviews_chart, use_container_width=True)
            
            with col4:
                # Reviews per Month Distribution - all data (focused on 1-10)
                if 'reviews_per_month' in review_df.columns and not review_df['reviews_per_month'].isna().all():
                    # Filter to 1-10 range and create explicit bins
                    reviews_filtered = review_df[(review_df['reviews_per_month'] >= 1) & (review_df['reviews_per_month'] <= 10)].copy()
                    
                    if not reviews_filtered.empty:
                        reviews_per_month_hist = alt.Chart(reviews_filtered).mark_bar().encode(
                            alt.X("reviews_per_month:Q", 
                                  bin=alt.Bin(extent=[1, 10], step=1, nice=False), 
                                  title="Reviews per Month",
                                  axis=alt.Axis(labelAngle=0)),
                            alt.Y("count()", title="Number of Listings"),
                            color=alt.value("#FFA500")
                        ).properties(
                            title="Reviews per Month Distribution (1-10 Reviews per Month)",
                            width=400,
                            height=300
                        )
                        st.altair_chart(reviews_per_month_hist, use_container_width=True)
                    else:
                        st.info("No data available in the 1-10 reviews per month range.")
except Exception as e:
    st.error(f"Error displaying review analysis: {str(e)}")

# --- PROPERTY CHARACTERISTICS SECTION ---
try:
    with st.expander("🏠 Property Characteristics", expanded=True):
        # Property Characteristics uses all quarters
        with st.spinner("Loading property characteristics data from all quarters..."):
            prop_filters = st.session_state.insights_filters.copy()
            prop_filters['quarters'] = []  # All quarters
            prop_df = query_insights_data(prop_filters, TABLE_CONFIG['all_quarters'], sample_size)
        
        if prop_df.empty:
            st.warning("No property characteristics data found.")
        else:
            col1, col2 = st.columns(2)
        
            with col1:
                # Bedroom Distribution
                if 'bedrooms' in prop_df.columns and not prop_df['bedrooms'].isna().all():
                    bedroom_counts = prop_df['bedrooms'].value_counts().sort_index().reset_index()
                    bedroom_counts.columns = ['bedrooms', 'count']
                    bedroom_chart = alt.Chart(bedroom_counts).mark_bar().encode(
                        alt.X("bedrooms:Q", title="Bedrooms"),
                        alt.Y("count:Q", title="Number of Listings"),
                        color=alt.value("#2196F3")
                    ).properties(
                        title="Bedroom Distribution",
                        width=400,
                        height=300
                    )
                    st.altair_chart(bedroom_chart, use_container_width=True)
            
            with col2:
                # Bathroom Distribution
                if 'bathrooms' in prop_df.columns and not prop_df['bathrooms'].isna().all():
                    bathroom_hist = alt.Chart(prop_df).mark_bar().encode(
                        alt.X("bathrooms:Q", bin=alt.Bin(maxbins=20), title="Bathrooms"),
                        alt.Y("count()", title="Number of Listings"),
                        color=alt.value("#2196F3")
                    ).properties(
                        title="Bathroom Distribution",
                        width=400,
                        height=300
                    )
                    st.altair_chart(bathroom_hist, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                # Accommodates Distribution
                if 'accommodates' in prop_df.columns and not prop_df['accommodates'].isna().all():
                    accommodates_counts = prop_df['accommodates'].value_counts().sort_index().reset_index()
                    accommodates_counts.columns = ['accommodates', 'count']
                    accommodates_chart = alt.Chart(accommodates_counts).mark_bar().encode(
                        alt.X("accommodates:Q", title="Accommodates"),
                        alt.Y("count:Q", title="Number of Listings"),
                        color=alt.value("#2196F3")
                    ).properties(
                        title="Accommodates Distribution",
                        width=400,
                        height=300
                    )
                    st.altair_chart(accommodates_chart, use_container_width=True)
            
            with col4:
                # Property Type Breakdown
                if 'property_category' in prop_df.columns and not prop_df['property_category'].isna().all():
                    property_type_counts = prop_df['property_category'].value_counts().reset_index()
                    property_type_counts.columns = ['property_type', 'count']
                    
                    # Calculate percentages
                    total_count = property_type_counts['count'].sum()
                    property_type_counts['percentage'] = (property_type_counts['count'] / total_count * 100).round(1)
                    property_type_counts['percentage_label'] = property_type_counts['percentage'].apply(lambda x: f"{x:.1f}%")
                    
                    # Create color scale with dark green for "Entire House"
                    # Get unique property types
                    unique_types = sorted(property_type_counts['property_type'].unique())
                    
                    # Build color range - assign dark green to "Entire House", others get default colors
                    color_range = []
                    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    color_idx = 0
                    for prop_type in unique_types:
                        if 'Entire House' in str(prop_type) or 'House' in str(prop_type):
                            color_range.append('#006400')  # Dark green
                        else:
                            color_range.append(default_colors[color_idx % len(default_colors)])
                            color_idx += 1
                    
                    # Sort by count for consistent ordering
                    property_type_counts = property_type_counts.sort_values('count', ascending=False).reset_index(drop=True)
                    
                    # Calculate midpoint theta (normalized 0-1) for each segment
                    property_type_counts['cumulative'] = property_type_counts['count'].cumsum()
                    property_type_counts['start_theta'] = (property_type_counts['cumulative'] - property_type_counts['count']) / total_count
                    property_type_counts['end_theta'] = property_type_counts['cumulative'] / total_count
                    property_type_counts['mid_theta'] = (property_type_counts['start_theta'] + property_type_counts['end_theta']) / 2
                    
                    # Create pie chart (full circle, no inner radius)
                    pie_chart = alt.Chart(property_type_counts).mark_arc(innerRadius=0).encode(
                        theta=alt.Theta("count:Q", stack=True),
                        color=alt.Color("property_type:N", 
                                      title="Property Type",
                                      scale=alt.Scale(
                                          domain=unique_types,
                                          range=color_range
                                      )),
                        tooltip=["property_type:N", alt.Tooltip("count:Q", format=","), alt.Tooltip("percentage:Q", format=".1f", title="Percentage (%)")]
                    )
                    
                    # Create labels using mark_text with theta encoding
                    labels = alt.Chart(property_type_counts).mark_text(
                        align='center',
                        baseline='middle',
                        fontSize=11,
                        fontWeight='bold',
                        color='white',
                        radius=120
                    ).encode(
                        theta=alt.Theta("mid_theta:Q"),
                        text="percentage_label:N"
                    )
                    
                    property_type_chart = (pie_chart + labels).properties(
                        title="Property Type Breakdown",
                        width=400,
                        height=300
                    )
                    st.altair_chart(property_type_chart, use_container_width=True)
            
            # Accommodates per Bedroom
            if 'accommodates_per_bedroom' in prop_df.columns and not prop_df['accommodates_per_bedroom'].isna().all():
                accommodates_per_bedroom_hist = alt.Chart(prop_df).mark_bar().encode(
                    alt.X("accommodates_per_bedroom:Q", bin=alt.Bin(maxbins=20), title="Accommodates per Bedroom"),
                    alt.Y("count()", title="Number of Listings"),
                    color=alt.value("#2196F3")
                ).properties(
                    title="Accommodates per Bedroom Distribution",
                    width=800,
                    height=300
                )
                st.altair_chart(accommodates_per_bedroom_hist, use_container_width=True)
except Exception as e:
    st.error(f"Error displaying property characteristics: {str(e)}")

# --- LOCATION INSIGHTS SECTION ---
try:
    with st.expander("📍 Location Insights", expanded=True):
        # Get data for all quarters to show quarter-over-quarter trends
        with st.spinner("Loading location data from all quarters..."):
            location_filters = st.session_state.insights_filters.copy()
            location_filters['quarters'] = []  # All quarters for trends
            location_df = query_temporal_data(location_filters, TABLE_CONFIG['all_quarters'], sample_size)
        
        if location_df.empty:
            st.warning("No location data found.")
        else:
            # Ensure quarter is numeric
            if 'quarter' in location_df.columns:
                location_df['quarter'] = pd.to_numeric(location_df['quarter'], errors='coerce')
            
            # Price by Location - Top 5 locations by average price across all quarters
            if 'src_location' in location_df.columns and 'price' in location_df.columns and 'quarter' in location_df.columns:
                # Calculate average price per location across all quarters
                location_avg_price = location_df.groupby('src_location')['price'].mean().reset_index()
                location_avg_price = location_avg_price.sort_values('price', ascending=False)
                
                # Get top 5 locations
                top_5_locations = location_avg_price.head(5)['src_location'].tolist()
                
                # Calculate average price per location per quarter
                location_quarter_price = location_df[location_df['src_location'].isin(top_5_locations)].groupby(['src_location', 'quarter'])['price'].mean().reset_index()
                location_quarter_price = location_quarter_price.sort_values(['src_location', 'quarter'])
                
                # Format price as integer for labels
                location_quarter_price['price_int'] = location_quarter_price['price'].astype(int)
                location_quarter_price['price_label'] = location_quarter_price['price_int'].astype(str)
                
                # Create line chart for top 5 locations
                chart = alt.Chart(location_quarter_price).mark_line(point=True, strokeWidth=3).encode(
                    alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                    alt.Y("price:Q", title="Average Price ($)"),
                    color=alt.Color("src_location:N", title="Location", scale=alt.Scale(scheme='category10')),
                    tooltip=["src_location:N", "quarter:O", alt.Tooltip("price:Q", format="$.3f", title="Average Price")]
                )
                
                # Add text labels at each point (as integers)
                labels = alt.Chart(location_quarter_price).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-10,
                    fontSize=10,
                    fontWeight='bold'
                ).encode(
                    alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                    alt.Y("price:Q", title="Average Price ($)"),
                    text="price_label:N",
                    color=alt.Color("src_location:N", scale=alt.Scale(scheme='category10'))
                )
                
                location_trend_chart = (chart + labels).properties(
                    title="Top 5 Locations by Average Price - Quarter over Quarter Trends",
                    width=800,
                    height=400
                )
                st.altair_chart(location_trend_chart, use_container_width=True)
except Exception as e:
    st.error(f"Error displaying location insights: {str(e)}")

# --- TEMPORAL TRENDS SECTION ---
try:
    with st.expander("📅 Quarterly Trends", expanded=True):
        # Temporal Trends uses all quarters - use a special query to ensure all quarters are represented
        with st.spinner("Loading temporal trends data from all quarters..."):
            # Create a fresh filter dict without quarter restrictions
            temporal_filters = {
                'locations': st.session_state.insights_filters.get('locations', []),
                'property_types': st.session_state.insights_filters.get('property_types', []),
                'superhost': st.session_state.insights_filters.get('superhost', 'All'),
                'quarters': []  # Explicitly set to empty to get all quarters
            }
            # Use a special query that samples from each quarter separately to ensure all quarters are represented
            temporal_df = query_temporal_data(temporal_filters, TABLE_CONFIG['all_quarters'], sample_size)
        
        if temporal_df.empty:
            st.warning("No temporal trends data found.")
        else:
            # Debug: Show available quarters and row count
            if 'quarter' in temporal_df.columns:
                available_quarters = sorted(temporal_df['quarter'].dropna().unique())
                quarter_counts = temporal_df['quarter'].value_counts().sort_index()            
            # Ensure quarter is numeric for proper sorting
            if 'quarter' in temporal_df.columns:
                temporal_df['quarter'] = pd.to_numeric(temporal_df['quarter'], errors='coerce')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price Trends Over Time - Line Chart with point labels
                if 'quarter' in temporal_df.columns and 'price' in temporal_df.columns:
                    price_by_quarter = temporal_df.groupby('quarter')['price'].mean().reset_index()
                    # Ensure quarters are sorted 1-4
                    price_by_quarter = price_by_quarter.sort_values('quarter')
                    # Format price for display
                    price_by_quarter['price_label'] = price_by_quarter['price'].apply(lambda x: f"${x:.0f}")
                    
                    # Create line chart
                    line = alt.Chart(price_by_quarter).mark_line(point=True, strokeWidth=3).encode(
                        alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                        alt.Y("price:Q", title="Average Price ($)"),
                        color=alt.value("#9C27B0"),
                        tooltip=["quarter:O", alt.Tooltip("price:Q", format="$.3f", title="Average Price")]
                    )
                    
                    # Add text labels at each point
                    labels = alt.Chart(price_by_quarter).mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-10,
                        fontSize=12,
                        fontWeight='bold'
                    ).encode(
                        alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                        alt.Y("price:Q", title="Average Price ($)"),
                        text="price_label:N",
                        color=alt.value("#9C27B0")
                    )
                    
                    price_trend = (line + labels).properties(
                        title="Price Trends by Quarter",
                        width=400,
                        height=300
                    )
                    st.altair_chart(price_trend, use_container_width=True)
            
            with col2:
                # Review Trends by Quarter - Line Chart with point labels
                if 'quarter' in temporal_df.columns and 'reviews_per_month' in temporal_df.columns:
                    reviews_by_quarter = temporal_df.groupby('quarter')['reviews_per_month'].mean().reset_index()
                    # Ensure quarters are sorted 1-4
                    reviews_by_quarter = reviews_by_quarter.sort_values('quarter')
                    # Format reviews for display
                    reviews_by_quarter['reviews_label'] = reviews_by_quarter['reviews_per_month'].apply(lambda x: f"{x:.3f}")
                    
                    # Create line chart
                    line = alt.Chart(reviews_by_quarter).mark_line(point=True, strokeWidth=3).encode(
                        alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                        alt.Y("reviews_per_month:Q", title="Average Reviews per Month"),
                        color=alt.value("#9C27B0"),
                        tooltip=["quarter:O", alt.Tooltip("reviews_per_month:Q", format=".3f", title="Average Reviews per Month")]
                    )
                    
                    # Add text labels at each point
                    labels = alt.Chart(reviews_by_quarter).mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-10,
                        fontSize=12,
                        fontWeight='bold'
                    ).encode(
                        alt.X("quarter:O", title="Quarter", sort=[1, 2, 3, 4], axis=alt.Axis(labelAngle=0)),
                        alt.Y("reviews_per_month:Q", title="Average Reviews per Month"),
                        text="reviews_label:N",
                        color=alt.value("#9C27B0")
                    )
                    
                    reviews_trend = (line + labels).properties(
                        title="Review Trends by Quarter",
                        width=400,
                        height=300
                    )
                    st.altair_chart(reviews_trend, use_container_width=True)
            
            # Instant Bookable Trends - Column Chart
            if 'quarter' in temporal_df.columns and 'instant_bookable' in temporal_df.columns:
                # Ensure quarter is numeric for proper sorting
                temporal_df['quarter'] = pd.to_numeric(temporal_df['quarter'], errors='coerce')
                
                instant_bookable_by_quarter = temporal_df.groupby(['quarter', 'instant_bookable']).size().reset_index(name='count')
                # Map 0/1 to No/Yes
                instant_bookable_by_quarter['instant_bookable_label'] = instant_bookable_by_quarter['instant_bookable'].map({0: 'No', 1: 'Yes'})
                
                # Calculate percentage of total for each quarter
                instant_bookable_by_quarter['percentage'] = instant_bookable_by_quarter.groupby('quarter')['count'].transform(lambda x: x / x.sum())
                
                # Ensure we have all quarters 1-4, even if some have no data
                all_quarters = [1, 2, 3, 4]
                existing_quarters = instant_bookable_by_quarter['quarter'].unique()
                missing_quarters = [q for q in all_quarters if q not in existing_quarters]
                
                # Sort by quarter
                instant_bookable_by_quarter = instant_bookable_by_quarter.sort_values('quarter')
                
                # Format percentage for labels
                instant_bookable_by_quarter['percentage_label'] = (instant_bookable_by_quarter['percentage'] * 100).apply(lambda x: f"{x:.1f}%")
                
                # Calculate cumulative percentage for label positioning (middle of each segment)
                instant_bookable_by_quarter = instant_bookable_by_quarter.sort_values(['quarter', 'instant_bookable'])
                instant_bookable_by_quarter['cumulative_pct'] = instant_bookable_by_quarter.groupby('quarter')['percentage'].cumsum()
                instant_bookable_by_quarter['label_position'] = instant_bookable_by_quarter['cumulative_pct'] - (instant_bookable_by_quarter['percentage'] / 2)
                
                # Create stacked bar chart
                bars = alt.Chart(instant_bookable_by_quarter).mark_bar().encode(
                    alt.X("quarter:O", title="Quarter", sort=all_quarters, axis=alt.Axis(labelAngle=0)),
                    alt.Y("percentage:Q", axis=alt.Axis(format=".0%"), title="Percentage of Listings"),
                    color=alt.Color("instant_bookable_label:N", title="Instant Bookable", scale=alt.Scale(domain=['No', 'Yes'], range=['#2196F3', '#4CAF50'])),
                    tooltip=["quarter", "instant_bookable_label", alt.Tooltip("count", format=","), alt.Tooltip("percentage", format=".1%")]
                )
                
                # Add percentage labels inside each segment
                labels = alt.Chart(instant_bookable_by_quarter).mark_text(
                    align='center',
                    baseline='middle',
                    fontSize=11,
                    fontWeight='bold',
                    color='white'
                ).encode(
                    alt.X("quarter:O", title="Quarter", sort=all_quarters, axis=alt.Axis(labelAngle=0)),
                    alt.Y("label_position:Q", axis=alt.Axis(format=".0%"), title="Percentage of Listings"),
                    text="percentage_label:N"
                )
                
                instant_bookable_trend = (bars + labels).properties(
                    title="Instant Bookable Trends by Quarter (All Quarters)",
                    width=800,
                    height=300
                )
                st.altair_chart(instant_bookable_trend, use_container_width=True)
except Exception as e:
    st.error(f"Error displaying temporal trends: {str(e)}")

# Download section removed per user request

