# us_rental_calculator
Based upon publicly available AirBnb US listings for selected markets

## Demo mode (no paid APIs)

The app runs without any API keys or cloud credentials. When a key is missing it
falls back to free sources:

| Feature | With keys | Without keys (demo mode) |
|---|---|---|
| Nearby listings map, Market Insights | BigQuery | Local snapshots in `data/` (latest-quarter listings; 20% sample for insights) |
| Address lookup | Mapbox | US Census Geocoder (free, no key); city center from the training data otherwise |
| Map tiles | Mapbox | Carto basemap (pydeck `road` style) |
| Mortgage rate | API Ninjas | Freddie Mac PMMS via FRED (free), dated snapshot as fallback |
| Property tax rate | API Ninjas | Tax Foundation county effective rates (`data/market_reference.json`) |
| Purchase price default | RentCast | Zillow Home Value Index for the market |
| Comparables | RentCast long-term rentals | Closest short-term rental listings with the same bedrooms and type |

`DATA_SOURCE=local` forces the local snapshots; `DATA_SOURCE=bigquery` requires BigQuery.

## Run with Docker

```bash
docker build -t pricision-ai .
docker run -p 8501:8501 pricision-ai
```

To use paid services, mount a secrets file at `/app/.streamlit/secrets.toml`.

## Deploy (Hostinger VPS with Traefik)

`deploy/docker-compose.yml` builds the image from this repository and exposes it
through the VPS's Traefik proxy with automatic HTTPS.
