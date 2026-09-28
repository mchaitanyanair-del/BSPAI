import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import scanpy as sc
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

# =====================================================================
# 1. FILE PATH CONFIGURATION & ANNDATA LOADING
# =====================================================================
DATA_DIR = r"C:\Users\Rishi\Downloads"

H5AD_PATH = os.path.join(DATA_DIR, "GSE131907_Lung_Cancer_preprocessed.h5ad")
SUPP_TABLE_1_PATH = os.path.join(
    DATA_DIR, "432_2024_5740_MOESM3_ESM (1).xlsx"
)  # Clinical labels table

print("=" * 65)
print("STEP 1: Loading Preprocessed scRNA-seq AnnData (.h5ad)...")
print("=" * 65)

if not os.path.exists(H5AD_PATH):
  raise FileNotFoundError(
      f"File not found: {H5AD_PATH}. Ensure the preprocessing script has"
      " finished."
  )

# Load primary AnnData object directly (fast binary load)
adata = sc.read_h5ad(H5AD_PATH)
print(
    f"Primary AnnData Ready: {adata.n_obs} single cells x {adata.n_vars} genes."
)

# Map uppercase gene names and common antibody target aliases
var_upper_map = {name.upper(): name for name in adata.var_names}
GENE_ALIASES = {
    "CD20": "MS4A1",
    "PD-1": "PDCD1",
    "PD1": "PDCD1",
    "PD-L1": "CD274",
    "PDL1": "CD274",
    "CTLA-4": "CTLA4",
    "BCMA": "TNFRSF17",
    "CD3": "CD3E",
    "C-MET": "MET",
    "HER2": "ERBB2",
    "HER3": "ERBB3",
    "4-1BB": "TNFRSF9",
}


def resolve_gene_symbol(gene_symbol):
  """Maps aliases and matches gene symbols against AnnData var_names."""
  s = str(gene_symbol).strip().upper()
  s = GENE_ALIASES.get(s, s)
  return var_upper_map.get(s, None)


# =====================================================================
# 2. BIOLOGICAL FEATURE EXTRACTION ENGINE
# =====================================================================
print("\n" + "=" * 65)
print("STEP 2: Initializing Biological Feature Extraction...")
print("=" * 65)

KEY_SUBTYPES = [
    "T:Treg",
    "T:Exhausted CD8+ T",
    "Myeloid:Macrophage",
    "Epithelial cells",
]


def extract_gene_vector(adata_sub, resolved_gene):
  """Extracts dense 1D expression vector for a gene in an AnnData subset."""
  if resolved_gene is not None and resolved_gene in adata_sub.var_names:
    vec = adata_sub[:, resolved_gene].X
    if hasattr(vec, "toarray"):
      vec = vec.toarray().flatten()
    return np.asarray(vec).flatten()
  return np.zeros(adata_sub.n_obs, dtype=np.float32)


def compute_pair_features(adata_obj, gene_a_str, gene_b_str, threshold=0.1):
  """Computes safe_avg, single/double positive ratios, and subtype correlations."""
  feats = {}
  ga = resolve_gene_symbol(gene_a_str)
  gb = resolve_gene_symbol(gene_b_str)

  # A. Safety Differential (safe_avg: tumor vs normal tissue differential)
  t_mask = adata_obj.obs["Cell_type.refined"].isin(["tLung", "Tumor"])
  n_mask = adata_obj.obs["Cell_type.refined"].isin(["nLung", "Normal"])

  va_t = extract_gene_vector(adata_obj[t_mask], ga)
  va_n = extract_gene_vector(adata_obj[n_mask], ga)
  vb_t = extract_gene_vector(adata_obj[t_mask], gb)
  vb_n = extract_gene_vector(adata_obj[n_mask], gb)

  d_a = float(np.mean(va_t) - np.mean(va_n))
  d_b = float(np.mean(vb_t) - np.mean(vb_n))

  # Harmonic mean formula: (d_a * d_b) / (d_a + d_b)
  if (d_a + d_b) != 0:
    feats["safe_avg"] = (d_a * d_b) / (d_a + d_b)
  else:
    feats["safe_avg"] = 0.0

  # B. Cell Subpopulation Activity & Single-Cell Pearson Correlations
  double_ratios = []
  single_ratios = []

  for subtype in KEY_SUBTYPES:
    clean_key = (
        subtype.replace(":", "_").replace("+", "plus").replace(" ", "_")
    )
    sub_mask = (
        adata_obj.obs["Cell_subtype"] == subtype
        if "Cell_subtype" in adata_obj.obs
        else np.zeros(adata_obj.n_obs, dtype=bool)
    )

    if np.sum(sub_mask) < 15:
      feats[f"corrcoef_{clean_key}"] = 0.0
      feats[f"double_ratio_{clean_key}"] = 0.0
      feats[f"sum_single_exp_{clean_key}"] = 0.0
      continue

    sub_adata = adata_obj[sub_mask]
    va = extract_gene_vector(sub_adata, ga)
    vb = extract_gene_vector(sub_adata, gb)

    # Pearson correlation r
    if np.std(va) > 1e-5 and np.std(vb) > 1e-5:
      r, _ = pearsonr(va, vb)
      feats[f"corrcoef_{clean_key}"] = 0.0 if np.isnan(r) else float(r)
    else:
      feats[f"corrcoef_{clean_key}"] = 0.0

    # Activity fractions above threshold
    pos_a = va > threshold
    pos_b = vb > threshold
    p_double = float(np.mean(pos_a & pos_b))
    p_single = float(np.mean(pos_a) + np.mean(pos_b))

    double_ratios.append(p_double)
    single_ratios.append(p_single)

    feats[f"double_ratio_{clean_key}"] = p_double
    feats[f"sum_single_exp_{clean_key}"] = float(np.mean(va) + np.mean(vb))

  feats["double_ratio_max"] = max(double_ratios) if double_ratios else 0.0
  feats["sum_single_ratio_max"] = max(single_ratios) if single_ratios else 0.0

  return feats


# =====================================================================
# 3. LOAD CLINICAL LABELS & PREPROCESS TARGET PAIRS
# =====================================================================
print("\n" + "=" * 65)
print("STEP 3: Loading Clinical Table & Engineering Features...")
print("=" * 65)

df_clin = pd.read_excel(SUPP_TABLE_1_PATH)
if "Drug Highest Phase" not in [str(c).strip() for c in df_clin.columns]:
  for r in range(1, 4):
    temp = pd.read_excel(SUPP_TABLE_1_PATH, header=r)
    if "Drug Highest Phase" in [str(c).strip() for c in temp.columns]:
      df_clin = temp
      break

df_clin.columns = df_clin.columns.astype(str).str.strip()

# Keep valid target pairs containing '+'
df_clin = df_clin[
    df_clin["Target(Gene Name)"].astype(str).str.contains(r"\+", na=False)
].copy()


def split_and_alphabetize_pair(target_str):
  """Splits targets and sorts alphabetically (ensures A+B == B+A)."""
  parts = [p.strip().upper() for p in str(target_str).split("+") if p.strip()]
  if len(parts) != 2:
    return pd.Series([np.nan, np.nan])
  parts.sort()
  return pd.Series([parts[0], parts[1]])


df_clin[["Target_A", "Target_B"]] = df_clin["Target(Gene Name)"].apply(
    split_and_alphabetize_pair
)
df_clin = df_clin.dropna(subset=["Target_A", "Target_B"]).reset_index(drop=True)

# 5-Tier Clinical Hierarchy Mapping from BSPAI study
phase_map = {
    "approved": 1,
    "nda": 1,
    "bla": 1,
    "phase 3": 2,
    "phase 2/3": 2,
    "phase 2": 3,
    "phase 1/2": 3,
    "phase 1": 3,
    "preclinical": 4,
    "ind": 4,
    "discontinued": 5,
    "pending": 5,
}


def map_clinical_rank(val):
  s = str(val).strip().lower()
  for k, v in phase_map.items():
    if k in s:
      return v
  return 4  # Default to preclinical if unassigned


df_clin["Clinical_Rank"] = df_clin["Drug Highest Phase"].apply(
    map_clinical_rank
)
# Binary anchor for evaluation: Approved / Late-Stage (Rank 1-2) vs Earlier Stages (Rank 3-5)
df_clin["Target_Label"] = (df_clin["Clinical_Rank"] <= 2).astype(int)

# Deduplicate identical pairs: retain highest achieved clinical milestone
unique_pairs = (
    df_clin.groupby(["Target_A", "Target_B"])
    .agg({"Clinical_Rank": "min", "Target_Label": "max"})
    .reset_index()
)

print(
    f"Calculating single-cell features for {len(unique_pairs)} unique target"
    " pairs..."
)

# Extract biological features across all unique pairs
feature_records = []
for idx, row in unique_pairs.iterrows():
  gA, gB = row["Target_A"], row["Target_B"]
  feats = compute_pair_features(adata, gA, gB)
  feats["Target_A"] = gA
  feats["Target_B"] = gB
  feats["Clinical_Rank"] = row["Clinical_Rank"]
  feats["Target_Label"] = row["Target_Label"]
  feature_records.append(feats)

feature_df = pd.DataFrame(feature_records)

# Feature matrix setup
meta_cols = ["Target_A", "Target_B", "Clinical_Rank", "Target_Label"]
feature_cols = [c for c in feature_df.columns if c not in meta_cols]

X = feature_df[feature_cols].copy()
y = feature_df["Target_Label"].copy()

pos_count = int((y == 1).sum())
neg_count = int((y == 0).sum())
scale_pos_weight = neg_count / max(pos_count, 1)

print("\n" + "=" * 65)
print("TRAINING DATASET SUMMARY")
print("=" * 65)
print(f"Total Unique Pairs Evaluated : {len(feature_df)}")
print(f"Validated Pairs (Class 1)   : {pos_count}")
print(f"Other / Early (Class 0)     : {neg_count}")
print(f"Scale Pos Weight            : {scale_pos_weight:.2f}")
print(f"Engineered Biological Feats : {len(feature_cols)}")

# =====================================================================
# 4. CROSS-VALIDATION & OUT-OF-FOLD EVALUATION (TESTING)
# =====================================================================
print("\n" + "=" * 65)
print("STEP 4: Cross-Validation & Model Evaluation...")
print("=" * 65)

# 5-Fold Stratified Cross-Validation
n_splits = min(5, max(2, pos_count))
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_probs = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
  X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
  X_va = X.iloc[val_idx]

  fold_model = xgb.XGBClassifier(
      n_estimators=100,
      max_depth=4,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      scale_pos_weight=scale_pos_weight,
      eval_metric="logloss",
      random_state=42,
  )
  fold_model.fit(X_tr, y_tr)
  oof_probs[val_idx] = fold_model.predict_proba(X_va)[:, 1]

cv_auc = roc_auc_score(y, oof_probs)
cv_pr_auc = average_precision_score(y, oof_probs)

# Calibrated classification threshold
custom_threshold = 0.35
oof_preds = (oof_probs >= custom_threshold).astype(int)

print(f"{n_splits}-Fold Cross-Validation ROC-AUC : {cv_auc:.4f}")
print(f"{n_splits}-Fold Cross-Validation PR-AUC  : {cv_pr_auc:.4f}")
print(f"\nOut-of-Fold Classification Report (Threshold={custom_threshold}):")
print(
    classification_report(
        y,
        oof_preds,
        target_names=["Other/Early", "Approved/Late"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
cm = confusion_matrix(y, oof_preds)
print(
    pd.DataFrame(
        cm,
        index=["Actual Other", "Actual Approved"],
        columns=["Pred Other", "Pred Approved"],
    )
)

# =====================================================================
# 5. HOLDOUT TRAIN / TEST SPLIT (20% Split)
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

holdout_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
holdout_model.fit(X_train, y_train)

y_test_proba = holdout_model.predict_proba(X_test)[:, 1]
y_test_preds = (y_test_proba >= custom_threshold).astype(int)

print("\n" + "-" * 50)
print("HOLDOUT TEST SET (20% Split) RESULTS")
print("-" * 50)
print(f"Test ROC-AUC : {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"Test PR-AUC  : {average_precision_score(y_test, y_test_proba):.4f}")
print(
    classification_report(
        y_test,
        y_test_preds,
        target_names=["Other/Early", "Approved/Late"],
        zero_division=0,
    )
)

# =====================================================================
# 6. TRAIN FINAL PRODUCTION MODEL
# =====================================================================
final_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
final_model.fit(X, y)
print("Production model trained on all extracted biological features.")

# Top Feature Importance Breakdown
importance = pd.DataFrame(
    {"Feature": feature_cols, "Score": final_model.feature_importances_}
).sort_values(by="Score", ascending=False)
print("\nTop 5 Most Informative Biological Features:")
print(importance.head(5).to_string(index=False))


# =====================================================================
# 7. LIVE INFERENCE ENGINE (PREDICTING ANY CANDIDATE PAIR)
# =====================================================================
def predict_target_pair(
    gene_a: str, gene_b: str, threshold: float = custom_threshold
):
  """Computes biological features directly from AnnData and predicts viability."""
  pair = sorted([gene_a.strip().upper(), gene_b.strip().upper()])
  live_feats = compute_pair_features(adata, pair[0], pair[1])
  input_df = pd.DataFrame([live_feats])[feature_cols]

  prob = float(final_model.predict_proba(input_df)[0][1])
  pred = int(prob >= threshold)

  print("\n" + "-" * 55)
  print(f"Candidate Pair                 : {pair[0]} + {pair[1]}")
  print(f"Tumor Safety Score (safe_avg)  : {live_feats['safe_avg']:.5f}")
  print(
      f"Max Double-Positive Ratio      : {live_feats['double_ratio_max']:.4f}"
  )
  print(f"Predicted Viability Probability: {prob:.4f}")
  print(
      f"Classification Verdict         : {'HIGH VIABILITY (Candidate Drug)' if pred == 1 else 'INVESTIGATIONAL / UNVIABLE'}"
  )
  print("-" * 55)
  return {"Pair": f"{pair[0]}+{pair[1]}", "Viability": prob, "Verdict": pred}


# =====================================================================
# 8. TEST BENCHMARK COMBINATIONS
# =====================================================================
print("\n" + "=" * 65)
print("STEP 5: Running Interactive Predictions...")
print("=" * 65)

# Known clinical archetypes
predict_target_pair("CD20", "CD3E")  # Approved T-cell engager (Glofitamab)
predict_target_pair("CD274", "CTLA4")  # Dual checkpoint inhibitor (AK104)
predict_target_pair("EGFR", "MET")  # Dual targeted kinase (Amivantamab)

# Novel combinations
predict_target_pair("TIGIT", "LAG3")  # Novel checkpoint pairing
predict_target_pair("CD40", "EGFR")  # Novel top candidate from Table 3
