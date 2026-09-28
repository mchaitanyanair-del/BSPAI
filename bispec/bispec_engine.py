"""
bspai_pipeline_unified.py
=============================================================================
Consolidated Machine Learning & Benchmarking Pipeline for BSPAI.
Integrates:
  1. Single-cell RNA-seq biological feature extraction (GSE131907).
  2. 5-tier clinical progression hierarchy mapping (Supplementary Table 1).
  3. Stratified 5-Fold Cross-Validation & Out-of-Fold Evaluation.
  4. Holdout Train/Test Split (80/20) validation.
  5. Dedicated Benchmarking against the published Top-100 Target Pairs.
  6. Artifact export for Streamlit (joblib model, features CSV, quick-slice h5ad).
  7. Live candidate inference engine with Table 5 discretized RAG payloads.
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import scanpy as sc
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

# =====================================================================
# 1. FILE PATH CONFIGURATION & CONSTANTS
# =====================================================================
DATA_DIR = r"C:\Users\Rishi\Downloads"

# Primary single-cell AnnData
H5AD_PATH = os.path.join(DATA_DIR, "GSE131907_Lung_Cancer_preprocessed.h5ad")

# 1. Training Labels: The 791 Clinical Drug Candidates (Supplementary Table 1)
SUPP_TABLE_1_PATH = os.path.join(DATA_DIR, "432_2024_5740_MOESM3_ESM (1).xlsx")

# 2. Benchmarking: Your separate dataset containing the Top 100 Published Targets

BENCHMARK_TABLE_PATH = r"C:\Users\Rishi\Downloads\432_2024_5740_MOESM5_ESM.xlsx"

# Export Artifacts for Streamlit Frontend
SAVED_FEATURES_CSV = os.path.join(DATA_DIR, "bspai_features.csv")
SAVED_MODEL_PATH = os.path.join(DATA_DIR, "bspai_model.joblib")
SAVED_SLICED_H5AD = os.path.join(DATA_DIR, "bspai_quick_slice.h5ad")

KEY_SUBTYPES = [
    "T:Treg",
    "T:Exhausted CD8+ T",
    "Myeloid:Macrophage",
    "Epithelial cells"
]

GENE_ALIASES = {
    "CD20": "MS4A1", "PD-1": "PDCD1", "PD1": "PDCD1", "PD-L1": "CD274",
    "PDL1": "CD274", "CTLA-4": "CTLA4", "BCMA": "TNFRSF17", "CD3": "CD3E",
    "C-MET": "MET", "HER2": "ERBB2", "HER3": "ERBB3", "4-1BB": "TNFRSF9"
}

# Table 5 Quantile Discretization Thresholds
DISCRETIZATION_BINS = {
    "safe_avg": [
        (-float("inf"), -0.0071, "Unsafe / Off-Tumor Risk"),
        (-0.0071, -0.000073, "Low Safety"),
        (-0.000073, 0.0012, "Safe / Tolerable"),
        (0.0012, 0.013, "Moderately High Safety"),
        (0.013, float("inf"), "High Safety")
    ],
    "double_ratio_max": [
        (0.0, 0.00019, "Low Co-expression"),
        (0.00019, 0.0015, "Moderately Low"),
        (0.0015, 0.0071, "Moderate Co-expression"),
        (0.0071, 0.35, "Above Average"),
        (0.35, 1.0, "High Co-localization")
    ]
}

def discretize_metric(metric_name: str, val: float) -> str:
    """Converts continuous biological metrics into qualitative categorical tags."""
    for low, high, tag in DISCRETIZATION_BINS.get(metric_name, []):
        if low <= val < high:
            return tag
    return "N/A"

# =====================================================================
# 2. ANNDATA LOADER & SYMBOL RESOLUTION
# =====================================================================
print("=" * 70)
print("STEP 1: Loading Single-Cell RNA-seq Matrix (AnnData)...")
print("=" * 70)

if not os.path.exists(H5AD_PATH):
    raise FileNotFoundError(f"AnnData file not found at: {H5AD_PATH}")

adata = sc.read_h5ad(H5AD_PATH)
print(f"Matrix loaded successfully: {adata.n_obs} cells x {adata.n_vars} genes.")

var_lookup_map = {name.upper(): name for name in adata.var_names}

def resolve_symbol(gene: str) -> str:
    clean = str(gene).strip().upper()
    clean = GENE_ALIASES.get(clean, clean)
    return var_lookup_map.get(clean, None)

def extract_dense_expression(sub_adata: sc.AnnData, resolved_gene: str) -> np.ndarray:
    if resolved_gene is not None and resolved_gene in sub_adata.var_names:
        v = sub_adata[:, resolved_gene].X
        if hasattr(v, "toarray"):
            v = v.toarray().flatten()
        return np.asarray(v).flatten()
    return np.zeros(sub_adata.n_obs, dtype=np.float32)

# =====================================================================
# 3. BIOLOGICAL FEATURE ENGINEERING ENGINE
# =====================================================================
def compute_pair_features(adata_obj: sc.AnnData, gene_a: str, gene_b: str, threshold: float = 0.1) -> dict:
    ga, gb = resolve_symbol(gene_a), resolve_symbol(gene_b)
    feats = {
        "Target_A_resolved": ga if ga else "Unresolved",
        "Target_B_resolved": gb if gb else "Unresolved"
    }

    # 1. Safety Differential: Harmonic mean across Tumor vs Normal Tissue
    t_mask = adata_obj.obs["Cell_type.refined"].isin(["tLung", "Tumor"])
    n_mask = adata_obj.obs["Cell_type.refined"].isin(["nLung", "Normal"])

    va_t, va_n = extract_dense_expression(adata_obj[t_mask], ga), extract_dense_expression(adata_obj[n_mask], ga)
    vb_t, vb_n = extract_dense_expression(adata_obj[t_mask], gb), extract_dense_expression(adata_obj[n_mask], gb)

    da = float(np.mean(va_t) - np.mean(va_n))
    db = float(np.mean(vb_t) - np.mean(vb_n))
    feats["safe_avg"] = (da * db) / (da + db) if (da + db) != 0 else 0.0

    # 2. Subpopulation Specific Co-expression & Pearson Correlations
    double_ratios = []
    single_ratios = []

    for sub in KEY_SUBTYPES:
        clean_key = sub.replace(":", "_").replace("+", "plus").replace(" ", "_")
        sub_mask = (adata_obj.obs["Cell_subtype"] == sub) if "Cell_subtype" in adata_obj.obs else np.zeros(adata_obj.n_obs, dtype=bool)

        if np.sum(sub_mask) < 15:
            feats[f"corrcoef_{clean_key}"] = 0.0
            feats[f"double_ratio_{clean_key}"] = 0.0
            feats[f"sum_single_exp_{clean_key}"] = 0.0
            continue

        sub_ad = adata_obj[sub_mask]
        va = extract_dense_expression(sub_ad, ga)
        vb = extract_dense_expression(sub_ad, gb)

        if np.std(va) > 1e-5 and np.std(vb) > 1e-5:
            r = pearsonr(va, vb)[0]
            feats[f"corrcoef_{clean_key}"] = 0.0 if np.isnan(r) else float(r)
        else:
            feats[f"corrcoef_{clean_key}"] = 0.0

        p_double = float(np.mean((va > threshold) & (vb > threshold)))
        p_single = float(np.mean(va > threshold) + np.mean(vb > threshold))

        double_ratios.append(p_double)
        single_ratios.append(p_single)

        feats[f"double_ratio_{clean_key}"] = p_double
        feats[f"sum_single_exp_{clean_key}"] = float(np.mean(va) + np.mean(vb))

    feats["double_ratio_max"] = max(double_ratios) if double_ratios else 0.0
    feats["sum_single_ratio_max"] = max(single_ratios) if single_ratios else 0.0
    return feats

# =====================================================================
# 4. CLINICAL DATA EXTRACTION & FEATURE MATRIX GENERATION
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Parsing Clinical Labels & Building Feature Matrix...")
print("=" * 70)

if not os.path.exists(SUPP_TABLE_1_PATH):
    raise FileNotFoundError(f"Clinical file not found at: {SUPP_TABLE_1_PATH}")

df_clin = pd.read_excel(SUPP_TABLE_1_PATH)
if "Drug Highest Phase" not in [str(c).strip() for c in df_clin.columns]:
    for r in range(1, 4):
        tmp = pd.read_excel(SUPP_TABLE_1_PATH, header=r)
        if "Drug Highest Phase" in [str(c).strip() for c in tmp.columns]:
            df_clin = tmp
            break

df_clin.columns = df_clin.columns.astype(str).str.strip()
df_clin = df_clin[df_clin["Target(Gene Name)"].astype(str).str.contains(r"\+", na=False)].copy()

def split_and_order_pair(s):
    pts = [p.strip().upper() for p in str(s).split("+") if p.strip()]
    if len(pts) != 2:
        return pd.Series([np.nan, np.nan])
    pts.sort()
    return pd.Series([pts[0], pts[1]])

df_clin[["Target_A", "Target_B"]] = df_clin["Target(Gene Name)"].apply(split_and_order_pair)
df_clin = df_clin.dropna(subset=["Target_A", "Target_B"]).reset_index(drop=True)

# 5-Tier Clinical Progression Hierarchy
phase_map = {
    "approved": 1, "nda": 1, "bla": 1,
    "phase 3": 2, "phase 2/3": 2,
    "phase 2": 3, "phase 1/2": 3, "phase 1": 3,
    "preclinical": 4, "ind": 4,
    "discontinued": 5, "pending": 5
}
df_clin["Clinical_Rank"] = df_clin["Drug Highest Phase"].apply(
    lambda x: next((v for k, v in phase_map.items() if k in str(x).lower()), 4)
)
# Rank 1-2 = Validated/Late Development, Rank 3-5 = Early/Other
df_clin["Target_Label"] = (df_clin["Clinical_Rank"] <= 2).astype(int)

# Deduplicate identical combinations by taking highest clinical progress
unique_pairs = (
    df_clin.groupby(["Target_A", "Target_B"])
    .agg({"Clinical_Rank": "min", "Target_Label": "max"})
    .reset_index()
)

print(f"Extracting biological features for {len(unique_pairs)} unique target pairs...")
records = []
for _, row in unique_pairs.iterrows():
    gA, gB = row["Target_A"], row["Target_B"]
    f = compute_pair_features(adata, gA, gB)
    f["Target_A"] = gA
    f["Target_B"] = gB
    f["Clinical_Rank"] = row["Clinical_Rank"]
    f["Target_Label"] = row["Target_Label"]
    records.append(f)

feature_df = pd.DataFrame(records)
meta_cols = ["Target_A", "Target_B", "Clinical_Rank", "Target_Label", "Target_A_resolved", "Target_B_resolved"]
feature_cols = [c for c in feature_df.columns if c not in meta_cols]

X = feature_df[feature_cols].copy()
y = feature_df["Target_Label"].copy()

pos_count = int((y == 1).sum())
neg_count = int((y == 0).sum())
scale_pos_weight = neg_count / max(pos_count, 1)

print(f"Dataset Built: {len(feature_df)} pairs | Positive: {pos_count} | Negative: {neg_count}")
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# =====================================================================
# 5. STRATIFIED CROSS-VALIDATION EVALUATION
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: Running Stratified 5-Fold Cross-Validation...")
print("=" * 70)

n_splits = min(5, max(2, pos_count))
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_probs = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va = X.iloc[val_idx]

    fold_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42
    )
    fold_model.fit(X_tr, y_tr)
    oof_probs[val_idx] = fold_model.predict_proba(X_va)[:, 1]

cv_auc = roc_auc_score(y, oof_probs)
cv_pr_auc = average_precision_score(y, oof_probs)
decision_threshold = 0.35
oof_preds = (oof_probs >= decision_threshold).astype(int)

print(f"{n_splits}-Fold Out-of-Fold ROC-AUC : {cv_auc:.4f}")
print(f"{n_splits}-Fold Out-of-Fold PR-AUC  : {cv_pr_auc:.4f}")
print(f"\nClassification Report (Threshold = {decision_threshold}):")
print(classification_report(y, oof_preds, target_names=["Early/Dev (0)", "Approved/Late (1)"], zero_division=0))
print("Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(y, oof_preds),
                   index=["Actual Dev", "Actual Approved"],
                   columns=["Pred Dev", "Pred Approved"]))

# =====================================================================
# 6. HOLDOUT TEST EVALUATION (80/20 SPLIT)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: Holdout Test Set Evaluation (20% Split)...")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
holdout_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", random_state=42
)
holdout_model.fit(X_train, y_train)

y_test_proba = holdout_model.predict_proba(X_test)[:, 1]
y_test_preds = (y_test_proba >= decision_threshold).astype(int)

print(f"Holdout Test ROC-AUC : {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"Holdout Test PR-AUC  : {average_precision_score(y_test, y_test_proba):.4f}")

# =====================================================================
# 7. FIT FINAL PRODUCTION MODEL & FEATURE IMPORTANCE
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Training Final Production Model...")
print("=" * 70)

final_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", random_state=42
)
final_model.fit(X, y)

feat_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": final_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("Top 5 Biological Features Driving Predictions:")
print(feat_imp.head(5).to_string(index=False))

# =====================================================================
# 8. BENCHMARK TESTING ON TOP-100 DATASET
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: Benchmarking Against Top 100 Published Targets...")
print("=" * 70)

if not os.path.exists(BENCHMARK_TABLE_PATH):
    print(f"[Notice] Benchmark file not found at: {BENCHMARK_TABLE_PATH}")
    print("Please set BENCHMARK_TABLE_PATH to your top-100 excel file path.")
else:
    try:
        bench_df = pd.read_excel(BENCHMARK_TABLE_PATH, header=1)
        bench_df.columns = bench_df.columns.astype(str).str.strip()

        # Handle header offset if columns are at row 0
        if "Gene1" not in bench_df.columns or "Gene2" not in bench_df.columns:
            bench_df = pd.read_excel(BENCHMARK_TABLE_PATH, header=0)
            bench_df.columns = bench_df.columns.astype(str).str.strip()

        if "Gene1" in bench_df.columns and "Gene2" in bench_df.columns:
            bench_results = []
            for _, row in bench_df.head(12).iterrows():
                g1 = str(row["Gene1"]).strip().upper()
                g2 = str(row["Gene2"]).strip().upper()

                # Calculate live biological features from AnnData
                p_feats = compute_pair_features(adata, g1, g2)
                score = float(final_model.predict_proba(pd.DataFrame([p_feats])[feature_cols])[0][1])

                bench_results.append({
                    "Target_Pair": f"{g1} + {g2}",
                    "Lit_Stage": row.get("clinical stage", "N/A"),
                    "Published_Score": row.get("predict score", np.nan),
                    "Our_Score": round(score, 4),
                    "Safe_Avg": round(p_feats["safe_avg"], 5),
                    "Double_Ratio_Max": round(p_feats["double_ratio_max"], 5)
                })

            df_summary = pd.DataFrame(bench_results)
            print(df_summary.to_string(index=False))

            # Calculate rank correlation if published scores exist
            valid_comp = df_summary.dropna(subset=["Published_Score"])
            if len(valid_comp) >= 5:
                corr, _ = spearmanr(valid_comp["Published_Score"], valid_comp["Our_Score"])
                print(f"\nSpearman Rank Correlation vs Published Scores: {corr:.4f}")
        else:
            print("Columns 'Gene1' and 'Gene2' not detected in benchmark table.")
    except Exception as err:
        print(f"Error during benchmark validation: {err}")

# =====================================================================
# 9. EXPORT STREAMLIT ARTIFACTS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: Exporting Cache Artifacts for Streamlit Frontend...")
print("=" * 70)

feature_df.to_csv(SAVED_FEATURES_CSV, index=False)
joblib.dump({"model": final_model, "feature_cols": feature_cols}, SAVED_MODEL_PATH)

sub_cells = (
    adata.obs["Cell_type.refined"].isin(["tLung", "Tumor", "nLung", "Normal"]) |
    adata.obs["Cell_subtype"].isin(KEY_SUBTYPES)
)
adata[sub_cells].copy().write_h5ad(SAVED_SLICED_H5AD)

print(f"Saved: {SAVED_FEATURES_CSV}")
print(f"Saved: {SAVED_MODEL_PATH}")
print(f"Saved: {SAVED_SLICED_H5AD}")

# =====================================================================
# 10. INTERACTIVE INFERENCE & RAG GENERATOR
# =====================================================================
def run_live_inference(gene_a: str, gene_b: str, threshold: float = decision_threshold) -> dict:
    pair = sorted([gene_a.strip().upper(), gene_b.strip().upper()])
    live_feats = compute_pair_features(adata, pair[0], pair[1])
    score = float(final_model.predict_proba(pd.DataFrame([live_feats])[feature_cols])[0][1])
    verdict = "APPROVED / CANDIDATE DRUG" if score >= threshold else "INVESTIGATIONAL / UNVIABLE"

    safe_tag = discretize_metric("safe_avg", live_feats["safe_avg"])
    double_tag = discretize_metric("double_ratio_max", live_feats["double_ratio_max"])

    rag_payload = f"""{{ml_result}}
1. Candidate Targets = [{pair[0]}] + [{pair[1]}]
2. Dual target expression double-positive score = [{double_tag}] ({live_feats['double_ratio_max']:.5f})
3. Target safety score = [{safe_tag}] ({live_feats['safe_avg']:.5f})
4. The final score of the machine learning model = [{score:.4f}]
5. Evaluated Microenvironments = T:Treg, T:Exhausted CD8+ T, Myeloid:Macrophage, Epithelial cells
"""
    return {
        "pair": f"{pair[0]} + {pair[1]}",
        "score": score,
        "verdict": verdict,
        "rag_payload": rag_payload
    }

print("\n" + "=" * 70)
print("STEP 8: Sample Live Candidate Inferences...")
print("=" * 70)

for ta, tb in [("CD20", "CD3E"), ("CD274", "CTLA4"), ("EGFR", "MET"), ("TIGIT", "LAG3"), ("CD40", "EGFR")]:
    res = run_live_inference(ta, tb)
    print(f"Pair: {res['pair']:<18} | Viability Score: {res['score']:.4f} | Verdict: {res['verdict']}")

print("\nPipeline execution complete.")
