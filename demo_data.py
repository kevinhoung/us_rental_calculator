"""
Free "demo mode" data sources for the public deployment.

When paid API keys or BigQuery credentials are not configured, the app falls
back to the helpers in this module:

- Nearby listings and Market Insights read local snapshots in data/
  (exported from the project's BigQuery tables).
- City coordinates, typical home values and property tax rates come from
  data/market_reference.json (sources listed inside the file).
- Street addresses are geocoded with the free US Census Geocoder (no key).
- Mortgage rates come from Freddie Mac's PMMS via FRED (no key), with a
  dated snapshot as fallback.

Set DATA_SOURCE=local to force local data even when BigQuery credentials
exist, or DATA_SOURCE=bigquery to require BigQuery.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"
NEARBY_FILE = DATA_DIR / "nearby_listings_q4.csv.gz"
INSIGHTS_FILE = DATA_DIR / "market_insights_sample20.csv.gz"
REFERENCE_FILE = DATA_DIR / "market_reference.json"

# Share of the full Market Insights table held in the local sample.
INSIGHTS_SAMPLE_PCT = 20

HTTP_HEADERS = {"User-Agent": "pricision-ai-demo/1.0"}


# --- Configuration -----------------------------------------------------------

def has_secret(section, key=None):
    """True if st.secrets has the section (and key), without raising when no secrets file exists."""
    try:
        if section not in st.secrets:
            return False
        if key is None:
            return True
        return bool(st.secrets[section].get(key))
    except Exception:
        return False


def use_local_data():
    """Use local snapshots instead of BigQuery."""
    source = os.environ.get("DATA_SOURCE", "auto").lower()
    if source == "local":
        return True
    if source == "bigquery":
        return False
    return not has_secret("gcp_service_account")


def is_demo_mode():
    """True when any paid service is unavailable, so the UI can label free fallbacks."""
    return use_local_data() or not (
        has_secret("mapbox", "token") and has_secret("rentcast", "api_key") and has_secret("api_ninjas", "api_key")
    )


# --- Market reference --------------------------------------------------------

@st.cache_resource
def load_market_reference():
    with open(REFERENCE_FILE) as f:
        return json.load(f)


def market_info(src_location):
    """Reference data for a training market (e.g. 'Los Angeles, CA'), or None."""
    return load_market_reference()["markets"].get(src_location)


def market_coordinates(src_location):
    info = market_info(src_location)
    if not info:
        return None
    return info["latitude"], info["longitude"]


def split_city_state(src_location):
    """'Los Angeles, CA' -> ('Los Angeles', 'CA')."""
    if ", " in src_location:
        city, state = src_location.rsplit(", ", 1)
        return city, state
    return src_location, None


# --- Geocoding ---------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def census_geocode(address):
    """
    Geocode a US street address with the US Census Geocoder (free, no key).
    Returns (latitude, longitude, city, state_abbr) or None.
    https://geocoding.geo.census.gov/geocoder/
    """
    try:
        resp = requests.get(
            "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
            params={"address": address, "benchmark": "Public_AR_Current", "format": "json"},
            headers=HTTP_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        matches = resp.json().get("result", {}).get("addressMatches", [])
        if not matches:
            return None
        best = matches[0]
        coords = best.get("coordinates", {})
        parts = best.get("addressComponents", {})
        city = (parts.get("city") or "").title() or None
        return float(coords["y"]), float(coords["x"]), city, parts.get("state")
    except Exception:
        return None


# --- Mortgage rates ----------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def _fred_latest(series_id):
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id},
        headers=HTTP_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df[pd.to_numeric(df.iloc[:, 1], errors="coerce").notna()]
    last = df.iloc[-1]
    return float(last.iloc[1]) / 100, str(last.iloc[0])


def mortgage_rate(loan_term):
    """
    Average US mortgage rate for the loan term (Freddie Mac PMMS via FRED).
    Terms of 15 years or less use the 15-year rate; longer terms use the 30-year rate.
    Returns (rate, as_of_date, label).
    """
    product = "15" if loan_term <= 15 else "30"
    series = "MORTGAGE15US" if product == "15" else "MORTGAGE30US"
    label = f"Freddie Mac {product}-year fixed average"
    try:
        rate, as_of = _fred_latest(series)
        return rate, as_of, label
    except Exception:
        snap = load_market_reference()["mortgage_rate_snapshot"]
        return snap[product], snap["as_of"], label + " (saved snapshot)"


# --- Nearby listings ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_nearby_listings():
    return pd.read_csv(NEARBY_FILE)


def _haversine_km(lat, lon, lats, lons):
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(lats), np.radians(lons)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def listings_within(latitude, longitude, radius_km):
    """All snapshot listings within radius_km, with a distance_km column."""
    df = load_nearby_listings()
    # Cheap bounding box before the exact distance.
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(np.cos(np.radians(latitude)), 0.01))
    box = df[
        df.latitude.between(latitude - dlat, latitude + dlat)
        & df.longitude.between(longitude - dlon, longitude + dlon)
    ].copy()
    box["distance_km"] = _haversine_km(latitude, longitude, box.latitude.values, box.longitude.values)
    return box[box.distance_km <= radius_km]


def nearby_listings_local(latitude, longitude, radius_km=5, limit=20):
    """Local equivalent of the BigQuery nearby-listings query."""
    near = listings_within(latitude, longitude, radius_km)
    if len(near) > limit:
        near = near.sample(n=limit, random_state=0)
    return near.drop(columns=["distance_km"]).reset_index(drop=True)


def comparable_listings(latitude, longitude, bedrooms, property_category, radius_km=3, n=10):
    """
    Closest listings in the snapshot with the same bedroom count and property type.
    Widens the radius once if fewer than n are found.
    """
    for radius in (radius_km, radius_km * 3):
        near = listings_within(latitude, longitude, radius)
        same = near[(near.bedrooms == bedrooms) & (near.property_category == property_category)]
        if len(same) >= n or radius != radius_km:
            break
    return same.sort_values("distance_km").head(n).reset_index(drop=True)


# --- Market Insights ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_insights_sample():
    return pd.read_csv(INSIGHTS_FILE)


def _apply_filters(df, filters_dict):
    if filters_dict.get("locations"):
        df = df[df.src_location.isin(filters_dict["locations"])]
    if filters_dict.get("property_types"):
        df = df[df.property_category.isin(filters_dict["property_types"])]
    if filters_dict.get("min_price") is not None:
        df = df[df.price >= filters_dict["min_price"]]
    if filters_dict.get("max_price") is not None:
        df = df[df.price <= filters_dict["max_price"]]
    if filters_dict.get("superhost", "All") != "All":
        df = df[df.host_is_superhost == (1 if filters_dict["superhost"] == "Yes" else 0)]
    return df


def _sample_fraction(sample_percent):
    # The local file already holds a 20% sample; scale the requested share against it.
    return min(1.0, sample_percent / INSIGHTS_SAMPLE_PCT)


def insights_data_local(filters_dict, sample_percent=10, max_rows=50000):
    """Local equivalent of query_insights_data()."""
    df = _apply_filters(load_insights_sample(), filters_dict)
    quarters = [int(q) for q in (filters_dict.get("quarters") or []) if q is not None]
    if quarters:
        df = df[df.quarter.isin(quarters)]
    frac = _sample_fraction(sample_percent)
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=0)
    return df.head(max_rows).reset_index(drop=True)


def temporal_data_local(filters_dict, sample_percent=10, rows_per_quarter=12500):
    """Local equivalent of query_temporal_data(): every quarter represented."""
    df = _apply_filters(load_insights_sample(), filters_dict)
    frac = _sample_fraction(sample_percent)
    parts = []
    for q in [1, 2, 3, 4]:
        part = df[df.quarter == q]
        if frac < 1.0:
            part = part.sample(frac=frac, random_state=0)
        parts.append(part.head(rows_per_quarter))
    return pd.concat(parts, ignore_index=True)


def filter_options_local():
    df = load_insights_sample()
    return (
        sorted(df.src_location.dropna().unique().tolist()),
        sorted(df.property_category.dropna().unique().tolist()),
        sorted(int(q) for q in df.quarter.dropna().unique()),
    )
