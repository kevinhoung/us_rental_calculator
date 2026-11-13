"""Pricision AI API.

Serves the nightly-rate model, nearby-listing comparables, market statistics,
free geocoding (Photon / US Census) and the weekly Freddie Mac mortgage rate.
No paid APIs and no credentials are required.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import pickle
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("pricision")
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
USER_AGENT = "PricisionAI/1.0 (short-term rental estimate demo)"

CATEGORIES = ["Entire Apartment/Condo", "Entire House", "Private Room", "Hotel/Resort"]
SCALING_FEATURES = [
    "latitude", "longitude", "review_scores_cleanliness", "review_scores_location",
    "instant_bookable", "log_beds_imputed", "log_accommodates",
    "log_bedrooms_imputed", "log_bathrooms_imputed",
]
CITY_LABELS = {
    "Washington Dc, DC": "Washington, DC",
    "Clark County, NV": "Las Vegas (Clark County), NV",
    "Hawaii, HI": "Hawaii (statewide), HI",
    "Rhode Island, RI": "Rhode Island (statewide), RI",
    "New York City, NY": "New York City, NY",
}

# --------------------------------------------------------------------------
# Load model and data once at startup
# --------------------------------------------------------------------------
with open(MODEL_DIR / "best_decisiontree_model.pkl", "rb") as f:
    MODEL = pickle.load(f)
SCALER = joblib.load(MODEL_DIR / "dt_standardscaler.joblib")
ENCODER = joblib.load(MODEL_DIR / "dt_encoder.joblib")
MODEL_COLUMNS = json.loads((MODEL_DIR / "dt_columns.json").read_text())
METRICS = json.loads((DATA_DIR / "model_metrics.json").read_text())
TAX = json.loads((DATA_DIR / "state_property_tax.json").read_text())

DF = pd.read_parquet(DATA_DIR / "listings.parquet")
DF = DF.dropna(subset=["lat", "lon", "price"]).reset_index(drop=True)
DF["city"] = DF["city"].astype(str)
DF["cat"] = DF["cat"].astype(str)
LAT = DF["lat"].to_numpy(np.float64)
LON = DF["lon"].to_numpy(np.float64)
PRICE = DF["price"].to_numpy(np.float64)

CITIES = (
    DF.groupby("city")
    .agg(lat=("lat", "median"), lon=("lon", "median"), listings=("price", "size"), median_price=("price", "median"))
    .reset_index()
)
CITIES["label"] = CITIES["city"].map(lambda c: CITY_LABELS.get(c, c))
CITIES["state"] = CITIES["city"].str[-2:]
CITIES = CITIES.sort_values("label").reset_index(drop=True)
log.info("Loaded %d listings across %d markets", len(DF), len(CITIES))

app = FastAPI(title="Pricision AI API", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
HTTP = httpx.Client(timeout=8.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearby_idx(lat: float, lon: float, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    box = np.nonzero((LAT > lat - dlat) & (LAT < lat + dlat) & (LON > lon - dlon) & (LON < lon + dlon))[0]
    if box.size == 0:
        return box, np.array([])
    d = haversine_km(lat, lon, LAT[box], LON[box])
    keep = d <= radius_km
    return box[keep], d[keep]


def coverage(lat: float, lon: float) -> dict:
    idx, dist = nearby_idx(lat, lon, 50.0)
    if idx.size == 0:
        return {"covered": False, "market": None, "label": None, "state": None, "nearest_km": None}
    j = idx[np.argmin(dist)]
    city = DF.at[j, "city"]
    return {
        "covered": bool(dist.min() <= 25.0),
        "market": city,
        "label": CITY_LABELS.get(city, city),
        "state": city[-2:],
        "nearest_km": round(float(dist.min()), 2),
    }


def pct(a: np.ndarray, q: float) -> Optional[float]:
    return float(np.percentile(a, q)) if a.size else None


def clean_float(x):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), 2)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class EstimateIn(BaseModel):
    lat: float = Field(ge=15, le=72)
    lon: float = Field(ge=-180, le=-60)
    category: Literal["Entire Apartment/Condo", "Entire House", "Private Room", "Hotel/Resort"]
    guests: int = Field(ge=1, le=16)
    bedrooms: int = Field(ge=0, le=10)
    beds: int = Field(ge=1, le=16)
    bathrooms: float = Field(ge=0.5, le=10)
    cleanliness: float = Field(default=4.8, ge=1, le=5)
    location_score: float = Field(default=4.8, ge=1, le=5)
    instant_book: bool = False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "listings": int(len(DF)), "markets": int(len(CITIES))}


@app.get("/api/meta")
def meta():
    return {
        "markets": CITIES[["city", "label", "state", "lat", "lon", "listings", "median_price"]].round(4).to_dict("records"),
        "categories": CATEGORIES,
        "listings": int(len(DF)),
        "metrics": METRICS,
        "data_source": "Inside Airbnb public listing snapshots (CC BY 4.0), four quarterly scrapes, cleaned",
    }


@app.post("/api/estimate")
def estimate(body: EstimateIn):
    bedrooms_for_model = max(body.bedrooms, 1)  # studios are treated as one bedroom by the model
    row = pd.DataFrame({
        "latitude": [body.lat],
        "longitude": [body.lon],
        "review_scores_cleanliness": [body.cleanliness],
        "review_scores_location": [body.location_score],
        "instant_bookable": [1.0 if body.instant_book else 0.0],
        "log_beds_imputed": [math.log(body.beds)],
        "log_accommodates": [math.log(body.guests)],
        "log_bedrooms_imputed": [math.log(bedrooms_for_model)],
        "log_bathrooms_imputed": [math.log(body.bathrooms)],
    })
    row[SCALING_FEATURES] = SCALER.transform(row[SCALING_FEATURES])
    ohe = ENCODER.transform(pd.DataFrame({"property_category": [body.category]}))
    ohe = ohe.toarray() if hasattr(ohe, "toarray") else ohe
    ohe_df = pd.DataFrame(ohe, columns=ENCODER.get_feature_names_out(["property_category"]))
    X = pd.concat([row, ohe_df], axis=1).reindex(columns=MODEL_COLUMNS, fill_value=0)
    price = float(np.exp(MODEL.predict(X)[0]))

    cov = coverage(body.lat, body.lon)

    # Comparables: everything within 5 km, and a "similar" subset (same type and bedrooms)
    radius = 5.0
    idx, dist = nearby_idx(body.lat, body.lon, radius)
    if idx.size < 30:
        radius = 15.0
        idx, dist = nearby_idx(body.lat, body.lon, radius)
    sub = DF.iloc[idx]
    prices = sub["price"].to_numpy(np.float64)
    sim_mask = (sub["cat"].to_numpy() == body.category) & (np.rint(sub["bed"].fillna(1).to_numpy()) == bedrooms_for_model)
    sim = prices[sim_mask]

    comps = {
        "radius_km": radius,
        "count": int(prices.size),
        "p25": pct(prices, 25), "median": pct(prices, 50), "p75": pct(prices, 75),
        "similar_count": int(sim.size),
        "similar_p25": pct(sim, 25), "similar_median": pct(sim, 50), "similar_p75": pct(sim, 75),
        "percentile_of_estimate": float((prices < price).mean() * 100) if prices.size else None,
    }

    # Map points: latest snapshot only, capped
    pts = sub.assign(dist=dist)
    if len(pts):
        pts = pts[pts["q"] == pts["q"].max()]
    if len(pts) > 400:
        pts = pts.sample(400, random_state=1)
    points = [
        {"lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5), "price": int(r.price), "cat": r.cat,
         "bedrooms": None if pd.isna(r.bed) else int(r.bed)}
        for r in pts.itertuples()
    ]

    state = cov["state"]
    return {
        "price": round(price, 2),
        "coverage": cov,
        "comps": comps,
        "points": points,
        "tax_rate_pct": TAX.get(state, TAX["_default"]) if state else TAX["_default"],
        "model_error_pct": METRICS["median_pct_error"],
    }


@app.get("/api/tax")
def tax(state: str = Query(min_length=2, max_length=2)):
    s = state.upper()
    return {"state": s, "rate_pct": TAX.get(s, TAX["_default"]), "approximate": True, "note": TAX["_note"]}


_RATE_CACHE: dict = {"t": 0.0, "v": None}


@app.get("/api/rates")
def rates():
    """Freddie Mac PMMS weekly averages, via the St. Louis Fed (FRED) public CSV."""
    if _RATE_CACHE["v"] and time.time() - _RATE_CACHE["t"] < 12 * 3600:
        return _RATE_CACHE["v"]
    try:
        r = HTTP.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "MORTGAGE30US,MORTGAGE15US"})
        r.raise_for_status()
        rows = [x for x in csv.reader(io.StringIO(r.text))][1:]
        rows = [x for x in rows if len(x) >= 3 and x[1] not in ("", ".") and x[2] not in ("", ".")]
        last = rows[-1]
        v = {"rate30": float(last[1]), "rate15": float(last[2]), "as_of": last[0],
             "source": "Freddie Mac Primary Mortgage Market Survey via FRED", "live": True}
    except Exception as e:  # noqa: BLE001
        log.warning("rate fetch failed: %s", e)
        if _RATE_CACHE["v"]:
            return _RATE_CACHE["v"]
        v = {"rate30": None, "rate15": None, "as_of": None, "source": None, "live": False}
        return v
    _RATE_CACHE.update(t=time.time(), v=v)
    return v


@lru_cache(maxsize=2048)
def _photon(q: str, limit: int) -> tuple:
    r = HTTP.get("https://photon.komoot.io/api/", params={"q": q, "limit": limit * 2, "lang": "en",
                                                          "bbox": "-179.9,18.5,-66.5,71.5"})
    r.raise_for_status()
    out = []
    for feat in r.json().get("features", []):
        p = feat.get("properties", {})
        if p.get("countrycode") not in (None, "US"):
            continue
        lon, lat = feat["geometry"]["coordinates"]
        street = " ".join(x for x in [p.get("housenumber"), p.get("street")] if x)
        name = p.get("name")
        first = street or name or ""
        parts = [first, p.get("city") or p.get("county"), p.get("state")]
        label = ", ".join(dict.fromkeys(x for x in parts if x))
        out.append({"label": label, "lat": lat, "lon": lon, "state": p.get("state"), "kind": p.get("type")})
        if len(out) >= limit:
            break
    return tuple(json.dumps(o) for o in out)


@app.get("/api/suggest")
def suggest(q: str = Query(min_length=3, max_length=120)):
    try:
        items = [json.loads(x) for x in _photon(q.strip().lower(), 6)]
    except Exception as e:  # noqa: BLE001
        log.warning("photon failed: %s", e)
        items = []
    for it in items:
        it["coverage"] = coverage(it["lat"], it["lon"])
    return {"items": items}


@lru_cache(maxsize=1024)
def _census(q: str) -> Optional[dict]:
    r = HTTP.get("https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
                 params={"address": q, "benchmark": "Public_AR_Current", "format": "json"})
    r.raise_for_status()
    m = r.json().get("result", {}).get("addressMatches", [])
    if not m:
        return None
    c = m[0]["coordinates"]
    return {"label": m[0]["matchedAddress"].title(), "lat": c["y"], "lon": c["x"]}


@app.get("/api/geocode")
def geocode(q: str = Query(min_length=2, max_length=160)):
    q = q.strip()
    hit = None
    if any(ch.isdigit() for ch in q):
        try:
            hit = _census(q)
        except Exception as e:  # noqa: BLE001
            log.warning("census failed: %s", e)
    if hit is None:
        try:
            items = [json.loads(x) for x in _photon(q.lower(), 1)]
            hit = items[0] if items else None
        except Exception as e:  # noqa: BLE001
            log.warning("photon failed: %s", e)
    if hit is None:
        # Last resort: match a covered market by name
        m = CITIES[CITIES["label"].str.lower().str.contains(q.lower().split(",")[0])]
        if len(m):
            hit = {"label": m.iloc[0]["label"], "lat": float(m.iloc[0]["lat"]), "lon": float(m.iloc[0]["lon"])}
    if hit is None:
        raise HTTPException(404, "We could not find that address. Try adding the city and state.")
    hit["coverage"] = coverage(hit["lat"], hit["lon"])
    return hit


# --------------------------------------------------------------------------
# Market insights
# --------------------------------------------------------------------------
@app.get("/api/market/filters")
def market_filters():
    return {
        "markets": CITIES[["city", "label"]].to_dict("records"),
        "categories": CATEGORIES,
        "quarters": sorted(int(x) for x in DF["q"].dropna().unique()),
    }


def _filter(city: str, cat: str, superhost: str, quarter: str, include_city: bool = True) -> pd.DataFrame:
    m = np.ones(len(DF), dtype=bool)
    if include_city and city != "all":
        m &= DF["city"].to_numpy() == city
    if cat != "all":
        m &= DF["cat"].to_numpy() == cat
    if superhost in ("yes", "no"):
        m &= (DF["sh"].fillna(0).to_numpy() == (1 if superhost == "yes" else 0))
    if quarter != "all":
        m &= DF["q"].fillna(0).to_numpy() == int(quarter)
    return DF[m]


@lru_cache(maxsize=512)
def _market(city: str, cat: str, superhost: str, quarter: str) -> str:
    d = _filter(city, cat, superhost, quarter)
    if d.empty:
        return json.dumps({"empty": True})
    p = d["price"].astype(float)
    sh = d["sh"].fillna(0)
    sh_med = p[sh == 1].median()
    non_med = p[sh == 0].median()

    by_city_df = _filter(city, cat, superhost, quarter, include_city=False)
    by_city = (by_city_df.groupby("city")["price"].median().sort_values(ascending=False).reset_index())
    by_city["label"] = by_city["city"].map(lambda c: CITY_LABELS.get(c, c))

    bins = list(range(0, 625, 25))
    counts, _ = np.histogram(p.clip(upper=624), bins=bins)
    hist = [{"from": int(bins[i]), "to": int(bins[i + 1]), "count": int(c)} for i, c in enumerate(counts)]
    over = int((p >= 600).sum())

    by_cat = d.groupby("cat")["price"].agg(["median", "size"]).reset_index().rename(columns={"size": "count"})
    bed = d.assign(b=d["bed"].fillna(1).clip(upper=5).round().astype(int)).groupby("b")["price"].agg(["median", "size"]).reset_index()
    by_q = d.groupby("q")["price"].median().reset_index()

    rbins = [0, 4.5, 4.7, 4.8, 4.9, 5.01]
    rlabels = ["Under 4.5", "4.5 to 4.7", "4.7 to 4.8", "4.8 to 4.9", "4.9 to 5.0"]
    rr = pd.cut(d["rating"], bins=rbins, labels=rlabels, right=False)
    by_rating = d.groupby(rr, observed=False)["price"].agg(["median", "size"]).reset_index()

    hosts = []
    for flag, name in ((1, "Superhost"), (0, "Other hosts")):
        h = d[sh == flag]
        hosts.append({"group": name, "count": int(len(h)), "median_price": clean_float(h["price"].median()),
                      "avg_rating": clean_float(h["rating"].mean()), "instant_book": clean_float(h["ib"].mean()),
                      "reviews_per_month": clean_float(h["rpm"].median())})

    points = []
    if city != "all":
        pts = d[d["q"] == d["q"].max()]
        if len(pts) > 1500:
            pts = pts.sample(1500, random_state=2)
        points = [[round(float(a), 4), round(float(b), 4), int(c)] for a, b, c in zip(pts["lat"], pts["lon"], pts["price"])]

    out = {
        "empty": False,
        "kpis": {
            "median": clean_float(p.median()), "p25": clean_float(p.quantile(0.25)), "p75": clean_float(p.quantile(0.75)),
            "count": int(len(d)), "superhost_share": clean_float(sh.mean()),
            "superhost_premium": clean_float(sh_med - non_med) if not (np.isnan(sh_med) or np.isnan(non_med)) else None,
            "avg_rating": clean_float(d["rating"].mean()),
        },
        "by_city": [{"city": r.city, "label": r.label, "median": clean_float(r.price)} for r in by_city.itertuples()],
        "histogram": hist, "over_600": over,
        "by_category": [{"cat": r.cat, "median": clean_float(r.median), "count": int(r.count)} for r in by_cat.itertuples()],
        "by_bedrooms": [{"bedrooms": int(r.b), "median": clean_float(r.median), "count": int(r.size)} for r in bed.itertuples()],
        "by_quarter": [{"quarter": int(r.q), "median": clean_float(r.price)} for r in by_q.itertuples()],
        "by_rating": [{"band": str(r.rating), "median": clean_float(r.median), "count": int(r.size)} for r in by_rating.itertuples()],
        "hosts": hosts,
        "points": points,
    }
    return json.dumps(out)


@app.get("/api/market")
def market(city: str = "all", cat: str = "all", superhost: str = "any", quarter: str = "all"):
    if city != "all" and city not in set(CITIES["city"]):
        raise HTTPException(400, "Unknown market")
    if cat != "all" and cat not in CATEGORIES:
        raise HTTPException(400, "Unknown property type")
    if quarter not in ("all", "1", "2", "3", "4"):
        raise HTTPException(400, "Unknown quarter")
    return json.loads(_market(city, cat, superhost, quarter))
