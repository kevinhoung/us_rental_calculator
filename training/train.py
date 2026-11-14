"""Retrain the nightly-price model with an honest held-out test set.

Listings repeat across the four quarterly scrapes and the data has no listing id,
so a random row split would leak. Rows are grouped by a pseudo listing id
(~500 m grid cell + property type + guests + bedrooms + beds + bathrooms) and
whole groups go to train or test. The same split scores the old decision tree.
"""
import json, pickle, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "api/data/listings.parquet")
df = df.dropna(subset=["lat", "lon", "price", "cat", "acc", "bed", "beds", "bath"])
df = df[(df.bath > 0) & (df.beds > 0) & (df.bed > 0) & (df.price > 0)].copy()
df["clean"] = df["clean"].fillna(df["clean"].median())
df["loc"] = df["loc"].fillna(df["loc"].median())
df["ib"] = df["ib"].fillna(0)
CATS = ["Entire Apartment/Condo", "Entire House", "Hotel/Resort", "Private Room"]

grp = (np.floor(df.lat.astype(float) / 0.005).astype(int).astype(str) + "_" +
       np.floor(df.lon.astype(float) / 0.005).astype(int).astype(str) + "_" +
       df.cat.astype(str) + "_" + df.acc.astype(str) + "_" + df.bed.astype(str) + "_" + df.beds.astype(str) + "_" + df.bath.astype(str))
tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, groups=grp))
tr, te = df.iloc[tr_idx], df.iloc[te_idx]
print(f"train {len(tr):,} rows, test {len(te):,} rows, groups {grp.nunique():,}")

def feats(d):
    return pd.DataFrame({
        "lat": d.lat.astype(float), "lon": d.lon.astype(float),
        "cat": pd.Categorical(d.cat.astype(str), categories=CATS).codes,
        "guests": d.acc.astype(float), "bedrooms": d.bed.astype(float), "beds": d.beds.astype(float),
        "bathrooms": d.bath.astype(float), "cleanliness": d.clean.astype(float),
        "location_score": d["loc"].astype(float), "instant_book": d.ib.astype(float),
    })

def score(y, p):
    ae = np.abs(p - y)
    return {"median_abs_error_usd": round(float(np.median(ae)), 1), "mean_abs_error_usd": round(float(ae.mean()), 1),
            "median_pct_error": round(float(np.median(ae / y)), 3), "share_within_25pct": round(float((ae / y <= 0.25).mean()), 3)}

y_tr, y_te = tr.price.astype(float).values, te.price.astype(float).values
Xtr, Xte = feats(tr), feats(te)
results = {}

# Baseline: city median from training rows
cm = tr.groupby("city", observed=True).price.median()
results["city_median"] = score(y_te, te.city.map(cm).astype(float).fillna(float(np.median(y_tr))).values)

# Old decision tree (trained on unknown split, so this flatters it if anything)
m_dir = ROOT  # original decision tree artifacts live at the repo root
dt = pickle.load(open(m_dir / "best_decisiontree_model.pkl", "rb"))
sc = joblib.load(m_dir / "dt_standardscaler.joblib"); enc = joblib.load(m_dir / "dt_encoder.joblib")
cols = json.load(open(m_dir / "dt_columns.json"))
X = pd.DataFrame({"latitude": te.lat.astype(float), "longitude": te.lon.astype(float), "review_scores_cleanliness": te.clean.astype(float),
                  "review_scores_location": te["loc"].astype(float), "instant_bookable": te.ib.astype(float),
                  "log_beds_imputed": np.log(te.beds.astype(float)), "log_accommodates": np.log(te.acc.astype(float)),
                  "log_bedrooms_imputed": np.log(te.bed.astype(float)), "log_bathrooms_imputed": np.log(te.bath.astype(float))})
X[list(sc.feature_names_in_)] = sc.transform(X[list(sc.feature_names_in_)])
oh = enc.transform(te[["cat"]].rename(columns={"cat": "property_category"}).astype(str))
oh = oh.toarray() if hasattr(oh, "toarray") else oh
X = pd.concat([X, pd.DataFrame(oh, columns=enc.get_feature_names_out(["property_category"]), index=X.index)], axis=1).reindex(columns=cols, fill_value=0)
results["old_decision_tree"] = score(y_te, np.exp(dt.predict(X)))

# New model: gradient boosting on log price
t = time.time()
hgb = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.08, max_iter=1500, max_leaf_nodes=255,
                                    min_samples_leaf=40, l2_regularization=1.0, categorical_features=[2],
                                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=40, random_state=0)
hgb.fit(Xtr, np.log(y_tr))
print(f"fit {time.time()-t:.0f}s, iters {hgb.n_iter_}")
results["gradient_boosting"] = score(y_te, np.exp(hgb.predict(Xte)))

# Quantile models for a per-property range (25th and 75th percentile)
qs = {}
for q in (0.25, 0.75):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.1, max_iter=600, max_leaf_nodes=127,
                                      min_samples_leaf=60, categorical_features=[2], early_stopping=True, random_state=0)
    m.fit(Xtr, np.log(y_tr)); qs[q] = m
lo, hi = np.exp(qs[0.25].predict(Xte)), np.exp(qs[0.75].predict(Xte))
results["range_coverage_25_75"] = round(float(((y_te >= lo) & (y_te <= hi)).mean()), 3)

print(json.dumps(results, indent=1))
out = ROOT / "api/model"
joblib.dump({"model": hgb, "q25": qs[0.25], "q75": qs[0.75], "features": list(Xtr.columns), "categories": CATS}, out / "hgb_model.joblib", compress=3)
json.dump(results, open(ROOT / "training/results.json", "w"), indent=1)
print("saved", (out / "hgb_model.joblib").stat().st_size / 1e6, "MB")
