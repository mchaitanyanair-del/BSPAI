"""
app.py - BSPAI Bispecific Target Discovery Studio
Combines precomputed clinical candidate benchmarks from bspai_features.csv
with instant single-cell lookups from bspai_quick_slice.h5ad.
"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
import scanpy as sc
import streamlit as st

# =====================================================================
# 1. PAGE SETUP & STYLING
# =====================================================================
st.set_page_config(
    page_title="BSPAI Discovery Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-container {
        border-radius: 8px;
        padding: 12px;
        background-color: #1e2124;
        border: 1px solid #2f3136;
    }
    .badge-approved {
        background-color: #2e7d32;
        color: #ffffff;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-unviable {
        background-color: #c62828;
        color: #ffffff;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. PATHS & QUANTIZATION CONSTANTS (Table 5)
# =====================================================================
DATA_DIR = r"C:\Users\Rishi\Downloads"
MODEL_PATH = os.path.join(DATA_DIR, "bspai_model.joblib")
SLICED_H5AD_PATH = os.path.join(DATA_DIR, "bspai_quick_slice.h5ad")
FEATURES_CSV_PATH = os.path.join(DATA_DIR, "bspai_features.csv")

KEY_SUBTYPES = ["T:Treg", "T:Exhausted CD8+ T", "Myeloid:Macrophage", "Epithelial cells"]

GENE_ALIASES = {
    "CD20": "MS4A1", "PD-1": "PDCD1", "PD1": "PDCD1", "PD-L1": "CD274",
    "PDL1": "CD274", "CTLA-4": "CTLA4", "BCMA": "TNFRSF17", "CD3": "CD3E",
    "C-MET": "MET", "HER2": "ERBB2", "HER3": "ERBB3", "4-1BB": "TNFRSF9"
}

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

def discretize_metric(metric_name: str, value: float) -> str:
    for low, high, tag in DISCRETIZATION_BINS.get(metric_name, []):
        if low <= value < high:
            return tag
    return "N/A"

# =====================================================================
# 3. CACHED ENGINE LOADER
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    # Verify cached resources exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SLICED_H5AD_PATH) or not os.path.exists(FEATURES_CSV_PATH):
        return None, None, None, None, None, None, False

    cached = joblib.load(MODEL_PATH)
    model = cached["model"]
    feature_cols = cached["feature_cols"]

    df_features = pd.read_csv(FEATURES_CSV_PATH)
    adata = sc.read_h5ad(SLICED_H5AD_PATH)
    var_map = {name.upper(): name for name in adata.var_names}

    # Pre-slice masks once for quick novel queries
    t_mask = adata.obs["Cell_type.refined"].isin(["tLung", "Tumor"])
    n_mask = adata.obs["Cell_type.refined"].isin(["nLung", "Normal"])
    sub_masks = {sub: (adata.obs["Cell_subtype"] == sub) for sub in KEY_SUBTYPES}

    return model, feature_cols, df_features, adata, var_map, (t_mask, n_mask, sub_masks), True

with st.spinner("Connecting to BSPAI Engine and cached single-cell artifacts..."):
    model, feature_cols, df_features, adata, var_map, masks, ready = load_all_artifacts()

if not ready:
    st.error("Cache artifacts not detected! Ensure `build_cache.py` has run in `C:\\Users\\Rishi\\Downloads`.")
    st.stop()

t_mask, n_mask, sub_masks = masks

# =====================================================================
# 4. SINGLE-CELL QUERY ENGINE (FOR CUSTOM PAIRS)
# =====================================================================
def resolve_symbol(s):
    clean = str(s).strip().upper()
    clean = GENE_ALIASES.get(clean, clean)
    return var_map.get(clean, None)

def extract_dense_vector(ad_sub, gene):
    if gene is not None and gene in ad_sub.var_names:
        v = ad_sub[:, gene].X
        if hasattr(v, "toarray"):
            v = v.toarray().flatten()
        return np.asarray(v).flatten()
    return np.zeros(ad_sub.n_obs, dtype=np.float32)

def compute_live_query(gene_a, gene_b):
    ga, gb = resolve_symbol(gene_a), resolve_symbol(gene_b)
    feats = {"Target_A": gene_a.upper(), "Target_B": gene_b.upper(), "ga_resolved": ga, "gb_resolved": gb}

    va_t, va_n = extract_dense_vector(adata[t_mask], ga), extract_dense_vector(adata[n_mask], ga)
    vb_t, vb_n = extract_dense_vector(adata[t_mask], gb), extract_dense_vector(adata[n_mask], gb)
    da = float(np.mean(va_t) - np.mean(va_n))
    db = float(np.mean(vb_t) - np.mean(vb_n))
    feats["safe_avg"] = (da * db) / (da + db) if (da + db) != 0 else 0.0

    double_ratios = []
    single_ratios = []
    for sub in KEY_SUBTYPES:
        k = sub.replace(":", "_").replace("+", "plus").replace(" ", "_")
        m = sub_masks[sub]
        if np.sum(m) < 15:
            feats[f"corrcoef_{k}"] = 0.0
            feats[f"double_ratio_{k}"] = 0.0
            feats[f"sum_single_exp_{k}"] = 0.0
            continue
        va, vb = extract_dense_vector(adata[m], ga), extract_dense_vector(adata[m], gb)
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
    return feats

# =====================================================================
# 5. SIDEBAR CONTROLS & BENCHMARK PAIR SELECTION
# =====================================================================
st.sidebar.title("🧬 BSPAI System")
st.sidebar.markdown("**Bispecific Antibody Target Predictor**\n*Powered by scRNA-seq microenvironment co-expression.*")
st.sidebar.markdown("---")

# Build target pair options from bspai_features.csv + archetypes
curated_archetypes = [
    "Approved: CD20 + CD3E (Glofitamab)",
    "Approved: CD274 + CTLA4 (Cadonilimab)",
    "Approved: EGFR + MET (Amivantamab)",
    "Novel Top Candidate: CD40 + EGFR (Table 3 #1)",
    "Novel Checkpoint: TIGIT + LAG3"
]

all_csv_pairs = [f"{r['Target_A']} + {r['Target_B']}" for _, r in df_features.iterrows()]
pair_choices = ["Custom Input"] + curated_archetypes + ["────────── All Database Pairs ──────────"] + all_csv_pairs

selected_option = st.sidebar.selectbox("Select Target Pair or Enter Custom", pair_choices, index=1)

# Populate Target A & Target B
if selected_option == "Custom Input" or "────" in selected_option:
    init_a, init_b = "CD274", "CTLA4"
elif "(" in selected_option:
    # Archetype string
    core = selected_option.split("(")[0].replace("Approved:", "").replace("Novel Top Candidate:", "").replace("Novel Checkpoint:", "").strip()
    init_a, init_b = [x.strip() for x in core.split("+")]
else:
    init_a, init_b = [x.strip() for x in selected_option.split("+")]

col_in_a, col_in_b = st.sidebar.columns(2)
target_a = col_in_a.text_input("Target A", value=init_a).strip().upper()
target_b = col_in_b.text_input("Target B", value=init_b).strip().upper()

decision_threshold = st.sidebar.slider("Viability Cutoff Threshold", 0.10, 0.90, 0.35, 0.05)
st.sidebar.markdown("---")
st.sidebar.caption(f"Database contains **{len(df_features)}** precomputed clinical drug pairs.")

# =====================================================================
# 6. INFERENCE & METRIC RETRIEVAL
# =====================================================================
st.title("Bispecific Target Viability Predictor")
st.caption("Predicting clinical development potential using single-cell transcriptomics (GSE131907)")

if st.button("Evaluate Candidate Combination", type="primary") or target_a:
    sorted_pair = sorted([target_a, target_b])
    
    # Check if pair is in precomputed bspai_features.csv
    match_mask = (df_features["Target_A"] == sorted_pair[0]) & (df_features["Target_B"] == sorted_pair[1])
    
    if match_mask.any():
        row = df_features[match_mask].iloc[0]
        feats = row.to_dict()
        input_df = pd.DataFrame([feats])[feature_cols]
        prob = float(model.predict_proba(input_df)[0][1])
        source_note = "Loaded from precomputed clinical database (bspai_features.csv)"
    else:
        feats = compute_live_query(sorted_pair[0], sorted_pair[1])
        input_df = pd.DataFrame([feats])[feature_cols]
        prob = float(model.predict_proba(input_df)[0][1])
        source_note = "Computed via real-time scRNA-seq array query (bspai_quick_slice.h5ad)"

    verdict = "APPROVED / CANDIDATE DRUG" if prob >= decision_threshold else "INVESTIGATIONAL / UNVIABLE"
    safe_tag = discretize_metric("safe_avg", feats.get("safe_avg", 0.0))
    double_tag = discretize_metric("double_ratio_max", feats.get("double_ratio_max", 0.0))

    # Top Metric Dashboard Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Viability Score", f"{prob:.4f}")
        if prob >= decision_threshold:
            st.markdown(f"<span class='badge-approved'>{verdict}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge-unviable'>{verdict}</span>", unsafe_allow_html=True)
    with c2:
        st.metric("Safety Score (safe_avg)", f"{feats.get('safe_avg', 0.0):.5f}", delta=safe_tag)
    with c3:
        st.metric("Max Double Positive", f"{feats.get('double_ratio_max', 0.0):.4f}", delta=double_tag)
    with c4:
        diff = prob - decision_threshold
        st.metric("Threshold Cutoff", f"{decision_threshold:.2f}", delta=f"{diff:+.2f} vs cutoff")

    st.caption(f"ℹ️ {source_note}")
    st.write("")

    # =====================================================================
    # 7. TME MICROENVIRONMENT CO-EXPRESSION & RAG PAYLOAD
    # =====================================================================
    tab_tme, tab_rag = st.tabs(["Microenvironment Co-expression", "Phase 2 RAG Prompt Payload"])

    with tab_tme:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Subpopulation Pearson Correlation (r)")
            r_vals = [
                feats.get("corrcoef_" + sub.replace(":", "_").replace("+", "plus").replace(" ", "_"), 0.0)
                for sub in KEY_SUBTYPES
            ]
            fig_radar = go.Figure(go.Scatterpolar(
                r=r_vals,
                theta=KEY_SUBTYPES,
                fill="toself",
                line_color="#1f77b4"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-0.2, 1.0])),
                height=340,
                margin=dict(l=30, r=30, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with chart_col2:
            st.subheader("Double-Positive Cellular Proportions")
            dp_vals = [
                feats.get("double_ratio_" + sub.replace(":", "_").replace("+", "plus").replace(" ", "_"), 0.0)
                for sub in KEY_SUBTYPES
            ]
            df_dp = pd.DataFrame({"Subpopulation": KEY_SUBTYPES, "Double-Positive Fraction": dp_vals})
            fig_bar = px.bar(
                df_dp,
                x="Subpopulation",
                y="Double-Positive Fraction",
                text_auto=".4f",
                color="Double-Positive Fraction",
                color_continuous_scale="Blues"
            )
            fig_bar.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_rag:
        st.subheader("Phase 2: RAG Context & LLM Prompt Payload")
        st.markdown(
            "Quantized biological measurements according to Table 5 from the study. "
            "This formatted text acts as `{ml_result}` to ground your upcoming RAG pipeline."
        )
        rag_payload = f"""{{ml_result}}
1. Candidate Targets = [{sorted_pair[0]}] + [{sorted_pair[1]}]
2. Dual target expression double-positive score = [{double_tag}] ({feats.get('double_ratio_max', 0.0):.5f})
3. Target safety score = [{safe_tag}] ({feats.get('safe_avg', 0.0):.5f})
4. The final score of the machine learning model = [{prob:.4f}]
5. Evaluated Microenvironments = T:Treg, T:Exhausted CD8+ T, Myeloid:Macrophage, Epithelial cells
"""
        st.code(rag_payload, language="markdown")

st.markdown("---")
st.caption("BSPAI Implementation — Non-Small Cell Lung Cancer Cohort (GSE131907) & Pairwise Progression Ranks.")
