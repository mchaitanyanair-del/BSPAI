# BSPAI: BiSpec Pairwise AI for Target Combination Prediction

Computational pipeline to predict, rank, and explain viable bispecific antibody (BsAb) target combinations using single-cell transcriptomics, biological graph ontologies, pairwise gradient boosting (XGBoost), and LLM-driven interpretation (GPT).

---

## 📌 Project Overview
Bispecific antibodies bind two distinct targets simultaneously to drive dual mechanisms of tumor suppression. However, identifying which target pairs are both therapeutically synergistic and safe is a major translational challenge. 

BSPAI frames target combination prediction as a **pairwise learning-to-rank** problem:
- **Phase 1 (Predictive Ranking):** Uses single-cell RNA-seq (scRNA-seq), Gene2Vec embeddings, and pathway co-occurrence data to train a pairwise XGBoost model against clinical progression stages.
- **Phase 2 (Contextual Explanation):** Maps machine learning predictions and discretized top features into an LLM prompt (RAG) to generate clinical analytical reports.

---

## 📁 Repository Structure
```text
├── Raw_Datasets/
│   ├── GSE131907_Lung_Cancer_cell_annotation.txt.gz  # scRNA-seq metadata (nLung, tLung, subtypes)
│   ├── GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz    # Raw count matrix (genes x cell barcodes)
│   ├── Supplementary_Table_1.xlsx                   # 791 clinical-stage bispecific drug labels
│   └── Supplementary_Table_3.xlsx                   # Model benchmark predictions (top-100 pairs)
├── data_preprocessing.py                            # Scanpy QC, normalization, and index alignment
├── feature_engineering.py                           # Computes safe_avg, corrcoef, double-positive ratios
├── model_training.py                                # Pairwise XGBoost (RankNet / rank:pairwise)
└── README.md
