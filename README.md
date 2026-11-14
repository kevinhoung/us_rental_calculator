# Pricision AI

Estimate what a US property should charge per night on Airbnb, then check whether the deal pays for itself.
Built on 883,120 public Airbnb listing snapshots (Inside Airbnb, CC BY 4.0) across 30 US markets.

## Architecture

| Part | Folder | Runs on | Notes |
|---|---|---|---|
| Front end | `web/` | Vercel (project `pricision-ai`) | Next.js 15, Tailwind 4, Recharts, MapLibre with OpenFreeMap tiles |
| API | `api/` | Hostinger VPS, Docker behind Traefik | FastAPI. Serves the model, comparables, market stats, geocoding and mortgage rates |

The browser calls the API directly at `NEXT_PUBLIC_API_BASE` (set in Vercel). No paid APIs or keys are needed:

- Model: `api/model/hgb_model.joblib` (scikit-learn gradient-boosting model plus quantile models for a likely price range)
- Listings: `api/data/listings.parquet`, exported from BigQuery table `US_all.us_master_cleaned_final_v2`
- Address search: Photon (OpenStreetMap) for autocomplete, US Census geocoder for full addresses
- Mortgage rate: Freddie Mac weekly survey via FRED
- Property tax: approximate statewide averages in `api/data/state_property_tax.json`

The original Streamlit app (`streamlit_app.py`, `pages/`) is kept for reference.

## Run locally

```bash
cd api && python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/uvicorn main:app --port 8000
```

```bash
cd web && npm install && npm run dev
```

The dev server proxies `/api` to `http://localhost:8000` unless `NEXT_PUBLIC_API_BASE` is set.

## Deploy

- **Front end:** Vercel builds `web/` (root directory setting). Env vars: `NEXT_PUBLIC_API_BASE`, `API_ORIGIN`.
- **API:** `deploy/api-compose.yml` is the Docker Compose project `pricision-api` on the VPS. The container downloads
  `main` from GitHub on start, so restarting the project deploys the latest commit. Served at
  `https://pricision-api.srv2145084.hstgr.cloud`.
