"""bspai_engine.py

Backend computational engine for BSPAI (BiSpec Pairwise AI).
Computes single-cell tumor microenvironment features and trains
the biological pairwise XGBoost viability model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import scanpy as sc
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

# =====================================================================
# 1. PATH CONFIGURATION & CONSTANTS
# =====================================================================
DATA_DIR = r"C:\Users\Rishi\Downloads"
H5AD_PATH = os.path.join(DATA_DIR, "GSE131907_Lung_Cancer_preprocessed.h5ad")
SUPP_TABLE_1_PATH = os.path.join(
    DATA_DIR, "432_2024_5740_MOESM3_ESM (1).xlsx"
)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "bspai_xgb_model.joblib")

KEY_SUBTYPES = [
    "T:Treg",
    "T:Exhausted CD8+ T",
    "Myeloid:Macrophage",
    "Epithelial cells",
]

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

# Quantile discretization thresholds from Table 5 in Zhang et al.
DISCRETIZATION_BINS = {
    "safe_avg": [
        (-float("inf"), -0.0071, "Unsafe / Off-Tumor Risk"),
        (-0.0071, -0.000073, "Low Safety"),
        (-0.000073, 0.0012, "Safe / Tolerable"),
        (0.0012, 0.013, "Moderately High Safety"),
        (0.013, float("inf"), "High Safety"),
    ],
    "double_ratio_max": [
        (0.0, 0.00019, "Low Co-expression"),
        (0.00019, 0.0015, "Moderately Low"),
        (0.0015, 0.0071, "Moderate Co-expression"),
        (0.0071, 0.35, "Above Average"),
        (0.35, 1.0, "High Co-localization"),
    ],
}


def discretize_metric(metric_name: str, value: float) -> str:
  """Converts continuous biological metrics into qualitative categorical tags."""
  if metric_name not in DISCRETIZATION_BINS:
    return "N/A"
  for low, high, tag in DISCRETIZATION_BINS[metric_name]:
    if low <= value < high:
      return tag
  return "N/A"


# =====================================================================
# 2. ANNDATA LOADER & SYMBOL RESOLUTION
# =====================================================================
def load_anndata(path: str = H5AD_PATH) -> sc.AnnData:
  """Loads the preprocessed single-cell RNA-seq object."""
  if not os.path.exists(path):
    raise FileNotFoundError(
        f"AnnData file not found at: {path}. Please verify the file path."
    )
  print(f"[Engine] Loading scRNA-seq matrix from {path}...")
  adata = sc.read_h5ad(path)
  print(f"[Engine] AnnData loaded: {adata.n_obs} cells x {adata.n_vars} genes.")
  return adata


def get_var_lookup_map(adata: sc.AnnData) -> dict:
  """Builds a case-insensitive lookup table for gene names in AnnData."""
  return {name.upper(): name for name in adata.var_names}


def resolve_gene_symbol(gene_symbol: str, var_upper_map: dict) -> str:
  """Resolves aliases and case-sensitivity against AnnData features."""
  s = str(gene_symbol).strip().upper()
  s = GENE_ALIASES.get(s, s)
  return var_upper_map.get(s, None)


def extract_gene_vector(
    adata_sub: sc.AnnData, resolved_gene: str
) -> np.ndarray:
  """Extracts a 1D dense expression array for a gene across an AnnData subset."""
  if resolved_gene is not None and resolved_gene in adata_sub.var_names:
    vec = adata_sub[:, resolved_gene].X
    if hasattr(vec, "toarray"):
      vec = vec.toarray().flatten()
    return np.asarray(vec).flatten()
  return np.zeros(adata_sub.n_obs, dtype=np.float32)


# =====================================================================
# 3. BIOLOGICAL FEATURE EXTRACTION
# =====================================================================
def compute_pair_features(
    adata_obj: sc.AnnData,
    gene_a_str: str,
    gene_b_str: str,
    threshold: float = 0.1,
) -> dict:
  """Computes safe_avg, single/double positive ratios, and subtype correlations."""
  var_lookup = get_var_lookup_map(adata_obj)
  ga = resolve_gene_symbol(gene_a_str, var_lookup)
  gb = resolve_gene_symbol(gene_b_str, var_lookup)

  feats = {
      "Target_A_resolved": ga if ga else "Unresolved",
      "Target_B_resolved": gb if gb else "Unresolved",
  }

  # 1. Safety differential: harmonic mean of (tumor - normal)
  t_mask = adata_obj.obs["Cell_type.refined"].isin(["tLung", "Tumor"])
  n_mask = adata_obj.obs["Cell_type.refined"].isin(["nLung", "Normal"])

  va_t = extract_gene_vector(adata_obj[t_mask], ga)
  va_n = extract_gene_vector(adata_obj[n_mask], ga)
  vb_t = extract_gene_vector(adata_obj[t_mask], gb)
  vb_n = extract_gene_vector(adata_obj[n_mask], gb)

  d_a = float(np.mean(va_t) - np.mean(va_n))
  d_b = float(np.mean(vb_t) - np.mean(vb_n))

  if (d_a + d_b) != 0:
    feats["safe_avg"] = (d_a * d_b) / (d_a + d_b)
  else:
    feats["safe_avg"] = 0.0

  # 2. Subpopulation-specific metrics
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

    # Pearson correlation
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
# 4. CLINICAL LABEL PARSING & DATASET BUILDER
# =====================================================================
def load_and_preprocess_clinical_labels(
    table_path: str = SUPP_TABLE_1_PATH,
) -> pd.DataFrame:
  """Loads Supplementary Table 1 and extracts standardized candidate pairs."""
  df_clin = pd.read_excel(table_path)
  if "Drug Highest Phase" not in [str(c).strip() for c in df_clin.columns]:
    for r in range(1, 4):
      temp = pd.read_excel(table_path, header=r)
      if "Drug Highest Phase" in [str(c).strip() for c in temp.columns]:
        df_clin = temp
        break

  df_clin.columns = df_clin.columns.astype(str).str.strip()
  df_clin = df_clin[
      df_clin["Target(Gene Name)"].astype(str).str.contains(r"\+", na=False)
  ].copy()

  def split_and_alphabetize_pair(target_str):
    parts = [p.strip().upper() for p in str(target_str).split("+") if p.strip()]
    if len(parts) != 2:
      return pd.Series([np.nan, np.nan])
    parts.sort()
    return pd.Series([parts[0], parts[1]])

  df_clin[["Target_A", "Target_B"]] = df_clin["Target(Gene Name)"].apply(
      split_and_alphabetize_pair
  )
  df_clin = df_clin.dropna(subset=["Target_A", "Target_B"]).reset_index(
      drop=True
  )

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
    return 4

  df_clin["Clinical_Rank"] = df_clin["Drug Highest Phase"].apply(
      map_clinical_rank
  )
  df_clin["Target_Label"] = (df_clin["Clinical_Rank"] <= 2).astype(int)

  unique_pairs = (
      df_clin.groupby(["Target_A", "Target_B"])
      .agg({"Clinical_Rank": "min", "Target_Label": "max"})
      .reset_index()
  )
  return unique_pairs


def build_feature_matrix(
    adata: sc.AnnData, unique_pairs: pd.DataFrame
) -> tuple:
  """Iterates over target pairs and builds the training feature matrix."""
  records = []
  print(
      f"[Engine] Computing biological features for {len(unique_pairs)} target"
      " pairs..."
  )
  for idx, row in unique_pairs.iterrows():
    gA, gB = row["Target_A"], row["Target_B"]
    feats = compute_pair_features(adata, gA, gB)
    feats["Target_A"] = gA
    feats["Target_B"] = gB
    feats["Clinical_Rank"] = row["Clinical_Rank"]
    feats["Target_Label"] = row["Target_Label"]
    records.append(feats)

  feature_df = pd.DataFrame(records)
  meta_cols = [
      "Target_A",
      "Target_B",
      "Clinical_Rank",
      "Target_Label",
      "Target_A_resolved",
      "Target_B_resolved",
  ]
  feature_cols = [c for c in feature_df.columns if c not in meta_cols]
  X = feature_df[feature_cols].copy()
  y = feature_df["Target_Label"].copy()
  return X, y, feature_cols, feature_df


# =====================================================================
# 5. MODEL TRAINING & INITIALIZATION
# =====================================================================
def train_production_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
  """Trains the final production gradient-boosted classifier."""
  pos_count = int((y == 1).sum())
  neg_count = int((y == 0).sum())
  scale_pos_weight = neg_count / max(pos_count, 1)

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
  return final_model


def get_trained_pipeline():
  """Loads resources and returns (adata, model, feature_cols, training_summary)."""
  adata = load_anndata(H5AD_PATH)
  unique_pairs = load_and_preprocess_clinical_labels(SUPP_TABLE_1_PATH)
  X, y, feature_cols, feature_df = build_feature_matrix(adata, unique_pairs)
  model = train_production_model(X, y)

  summary = {
      "total_pairs": len(feature_df),
      "pos_count": int((y == 1).sum()),
      "neg_count": int((y == 0).sum()),
      "feature_count": len(feature_cols),
  }
  return adata, model, feature_cols, summary


# =====================================================================
# 6. INFERENCE & RAG PAYLOAD GENERATOR
# =====================================================================
def run_live_inference(
    adata: sc.AnnData,
    model: xgb.XGBClassifier,
    feature_cols: list,
    gene_a: str,
    gene_b: str,
    threshold: float = 0.35,
) -> dict:
  """Predicts viability probability, extracts feature details, and prepares RAG payload."""
  pair = sorted([gene_a.strip().upper(), gene_b.strip().upper()])
  live_feats = compute_pair_features(adata, pair[0], pair[1])

  # Prepare numeric vector for inference
  input_df = pd.DataFrame([live_feats])[feature_cols]
  prob = float(model.predict_proba(input_df)[0][1])
  verdict = (
      "APPROVED / CANDIDATE DRUG"
      if prob >= threshold
      else "INVESTIGATIONAL / UNVIABLE"
  )

  # Discretize for natural language explanation
  safe_tag = discretize_metric("safe_avg", live_feats["safe_avg"])
  double_tag = discretize_metric(
      "double_ratio_max", live_feats["double_ratio_max"]
  )

  # Formatted prompt payload matching Phase 2 from the BSPAI study
  rag_payload = f"""{{ml_result}}
1. Candidate Targets = [{pair[0]}] + [{pair[1]}]
2. Dual target expression double-positive score = [{double_tag}] ({live_feats['double_ratio_max']:.5f})
3. Target safety score = [{safe_tag}] ({live_feats['safe_avg']:.5f})
4. The final score of the machine learning model = [{prob:.4f}]
5. Evaluated Microenvironments = T:Treg, T:Exhausted CD8+ T, Myeloid:Macrophage, Epithelial cells
"""

  return {
      "pair": f"{pair[0]} + {pair[1]}",
      "target_a": pair[0],
      "target_b": pair[1],
      "target_a_resolved": live_feats.get("Target_A_resolved", "N/A"),
      "target_b_resolved": live_feats.get("Target_B_resolved", "N/A"),
      "probability": prob,
      "verdict": verdict,
      "features": live_feats,
      "safe_tag": safe_tag,
      "double_tag": double_tag,
      "rag_payload": rag_payload,
  }


if __name__ == "__main__":
  # Standalone verification run
  adata, model, feature_cols, summary = get_trained_pipeline()
  print("\n[Engine Ready] Summary:", summary)
  sample_pred = run_live_inference(
      adata, model, feature_cols, "CD274", "CTLA4", threshold=0.35
  )
  print(f"[Verification Prediction] Score: {sample_pred['probability']:.4f}")
