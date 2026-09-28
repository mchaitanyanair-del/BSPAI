"""
build_cache.py - Run once to create precomputed cache files.
Execution time: ~1-2 minutes. After this, your app will load in < 1 second.
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import scanpy as sc
import xgboost as xgb

DATA_DIR = r"C:\Users\Rishi\Downloads"
H5AD_PATH = os.path.join(DATA_DIR, "GSE131907_Lung_Cancer_preprocessed.h5ad")
SUPP_TABLE_1_PATH = os.path.join(DATA_DIR, "432_2024_5740_MOESM3_ESM (1).xlsx")

FEATURES_CSV_PATH = os.path.join(DATA_DIR, "bspai_features.csv")
MODEL_PATH = os.path.join(DATA_DIR, "bspai_model.joblib")
SLICED_H5AD_PATH = os.path.join(DATA_DIR, "bspai_quick_slice.h5ad")

KEY_SUBTYPES = ["T:Treg", "T:Exhausted CD8+ T", "Myeloid:Macrophage", "Epithelial cells"]

GENE_ALIASES = {
    "CD20": "MS4A1", "PD-1": "PDCD1", "PD1": "PDCD1", "PD-L1": "CD274",
    "PDL1": "CD274", "CTLA-4": "CTLA4", "BCMA": "TNFRSF17", "CD3": "CD3E",
    "C-MET": "MET", "HER2": "ERBB2", "HER3": "ERBB3", "4-1BB": "TNFRSF9"
}

print("1. Loading full AnnData...")
adata = sc.read_h5ad(H5AD_PATH)
var_upper_map = {name.upper(): name for name in adata.var_names}

def resolve(g):
    s = str(g).strip().upper()
    s = GENE_ALIASES.get(s, s)
    return var_upper_map.get(s, None)

def get_vec(ad_sub, gene):
    if gene is not None and gene in ad_sub.var_names:
        v = ad_sub[:, gene].X
        if hasattr(v, "toarray"):
            v = v.toarray().flatten()
        return np.asarray(v).flatten()
    return np.zeros(ad_sub.n_obs, dtype=np.float32)

# Load Clinical Pairs
print("2. Parsing Clinical Table...")
df_clin = pd.read_excel(SUPP_TABLE_1_PATH)
if "Drug Highest Phase" not in [str(c).strip() for c in df_clin.columns]:
    for r in range(1, 4):
        temp = pd.read_excel(SUPP_TABLE_1_PATH, header=r)
        if "Drug Highest Phase" in [str(c).strip() for c in temp.columns]:
            df_clin = temp
            break
df_clin.columns = df_clin.columns.astype(str).str.strip()
df_clin = df_clin[df_clin["Target(Gene Name)"].astype(str).str.contains(r"\+", na=False)].copy()

def split_pair(s):
    pts = [p.strip().upper() for p in str(s).split("+") if p.strip()]
    if len(pts) != 2:
        return pd.Series([np.nan, np.nan])
    pts.sort()
    return pd.Series([pts[0], pts[1]])

df_clin[["Target_A", "Target_B"]] = df_clin["Target(Gene Name)"].apply(split_pair)
df_clin = df_clin.dropna(subset=["Target_A", "Target_B"]).reset_index(drop=True)

phase_map = {
    "approved": 1, "nda": 1, "bla": 1, "phase 3": 2, "phase 2/3": 2,
    "phase 2": 3, "phase 1/2": 3, "phase 1": 3, "preclinical": 4,
    "ind": 4, "discontinued": 5, "pending": 5
}
df_clin["Clinical_Rank"] = df_clin["Drug Highest Phase"].apply(
    lambda x: next((v for k, v in phase_map.items() if k in str(x).lower()), 4)
)
df_clin["Target_Label"] = (df_clin["Clinical_Rank"] <= 2).astype(int)

unique_pairs = df_clin.groupby(["Target_A", "Target_B"]).agg({"Clinical_Rank": "min", "Target_Label": "max"}).reset_index()

print("3. Pre-extracting single-cell feature matrix...")
t_mask = adata.obs["Cell_type.refined"].isin(["tLung", "Tumor"])
n_mask = adata.obs["Cell_type.refined"].isin(["nLung", "Normal"])

records = []
for _, row in unique_pairs.iterrows():
    ga, gb = resolve(row["Target_A"]), resolve(row["Target_B"])
    feats = {"Target_A": row["Target_A"], "Target_B": row["Target_B"], "Target_Label": row["Target_Label"]}
    
    va_t, va_n = get_vec(adata[t_mask], ga), get_vec(adata[n_mask], ga)
    vb_t, vb_n = get_vec(adata[t_mask], gb), get_vec(adata[n_mask], gb)
    da = float(np.mean(va_t) - np.mean(va_n))
    db = float(np.mean(vb_t) - np.mean(vb_n))
    feats["safe_avg"] = (da * db) / (da + db) if (da + db) != 0 else 0.0

    double_ratios = []
    single_ratios = []
    for sub in KEY_SUBTYPES:
        k = sub.replace(":", "_").replace("+", "plus").replace(" ", "_")
        s_mask = (adata.obs["Cell_subtype"] == sub) if "Cell_subtype" in adata.obs else np.zeros(adata.n_obs, dtype=bool)
        if np.sum(s_mask) < 15:
            feats[f"corrcoef_{k}"] = 0.0
            feats[f"double_ratio_{k}"] = 0.0
            feats[f"sum_single_exp_{k}"] = 0.0
            continue
        va, vb = get_vec(adata[s_mask], ga), get_vec(adata[s_mask], gb)
        r = pearsonr(va, vb)[0] if (np.std(va) > 1e-5 and np.std(vb) > 1e-5) else 0.0
        feats[f"corrcoef_{k}"] = 0.0 if np.isnan(r) else float(r)
        p_double = float(np.mean((va > 0.1) & (vb > 0.1)))
        p_single = float(np.mean(va > 0.1) + np.mean(vb > 0.1))
        double_ratios.append(p_double)
        single_ratios.append(p_single)
        feats[f"double_ratio_{k}"] = p_double
        feats[f"sum_single_exp_{k}"] = float(np.mean(va) + np.mean(vb))

    feats["double_ratio_max"] = max(double_ratios) if double_ratios else 0.0
    feats["sum_single_ratio_max"] = max(single_ratios) if single_ratios else 0.0
    records.append(feats)

feature_df = pd.DataFrame(records)
feature_df.to_csv(FEATURES_CSV_PATH, index=False)
print(f"Saved: {FEATURES_CSV_PATH}")

print("4. Training & caching XGBoost model...")
feat_cols = [c for c in feature_df.columns if c not in ["Target_A", "Target_B", "Target_Label"]]
X = feature_df[feat_cols]
y = feature_df["Target_Label"]
scale_pos_weight = int((y == 0).sum()) / max(int((y == 1).sum()), 1)

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", random_state=42
)
model.fit(X, y)
joblib.dump({"model": model, "feature_cols": feat_cols}, MODEL_PATH)
print(f"Saved: {MODEL_PATH}")

print("5. Saving subpopulation slice for live queries...")
# Keep only essential cell populations to make live lookups instant
sub_cells = adata.obs["Cell_type.refined"].isin(["tLung", "Tumor", "nLung", "Normal"]) | \
            adata.obs["Cell_subtype"].isin(KEY_SUBTYPES)
adata_sub = adata[sub_cells].copy()
adata_sub.write_h5ad(SLICED_H5AD_PATH)
print(f"Saved: {SLICED_H5AD_PATH}")
print("Done! You can now run 'streamlit run app.py' for near-instant execution.")
