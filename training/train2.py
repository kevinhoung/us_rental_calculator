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

from sklearn.neighbors import KDTree
K = 40
def xy(d):
    lat = d.lat.astype(float).values
    return np.c_[d.lon.astype(float).values * np.cos(np.radians(lat)), lat]

def nbr_feats(query_df, ref_df, exclude_same_group=False, q_groups=None, r_groups=None):
    """Median and 25/75th pct of log price of the K nearest reference listings (all types) and of the same type."""
    out = {}
    ref_lp = np.log(ref_df.price.astype(float).values)
    for name, mask_q, mask_r in [("all", None, None)] + [(c, (query_df.cat.astype(str) == c).values, (ref_df.cat.astype(str) == c).values) for c in CATS]:
        rsel = np.arange(len(ref_df)) if mask_r is None else np.nonzero(mask_r)[0]
        qsel = np.arange(len(query_df)) if mask_q is None else np.nonzero(mask_q)[0]
        if len(qsel) == 0:
            continue
        tree = KDTree(xy(ref_df.iloc[rsel]))
        dist, ind = tree.query(xy(query_df.iloc[qsel]), k=min(K + 10, len(rsel)))
        ind = rsel[ind]
        vals = ref_lp[ind]
        if exclude_same_group:
            vals = np.where(r_groups[ind] == q_groups[qsel][:, None], np.nan, vals)
        vals = vals[:, :K + 10]
        med = np.nanmedian(vals[:, :], axis=1)
        p25 = np.nanpercentile(vals, 25, axis=1); p75 = np.nanpercentile(vals, 75, axis=1)
        d = dist[:, K - 1] * 111.0
        key = "all" if name == "all" else "same"
        for col, v in (("med", med), ("p25", p25), ("p75", p75), ("dist", d)):
            out.setdefault(f"nb_{key}_{col}", np.full(len(query_df), np.nan))[qsel] = v
    return pd.DataFrame(out, index=query_df.index)

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
g_arr = grp.values
t0 = time.time()
NBtr = nbr_feats(tr, tr, exclude_same_group=True, q_groups=g_arr[tr_idx], r_groups=g_arr[tr_idx])
NBte = nbr_feats(te, tr)
print(f"neighbour features {time.time()-t0:.0f}s")
Xtr, Xte = pd.concat([feats(tr), NBtr], axis=1), pd.concat([feats(te), NBte], axis=1)
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
hgb = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.08, max_iter=3000, max_leaf_nodes=255,
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
