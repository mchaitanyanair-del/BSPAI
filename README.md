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



BI Spec Pairwise AI Completed Project — From Input to Final Output

What Is BI Spec Pairwise AI? 
BI Spec Pairwise AI is a software system that uses biological data and machine learning to analyse combinations of two biological targets that could be relevant to bispecific antibody development. A bispecific antibody is designed to interact with two different targets. The project therefore does not study only one target at a time. Its central unit of analysis is a pair of targets. The system represents that pair using biological and other engineered information and uses a trained machine-learning model to produce a computational result. The completed application brings together four major parts: the data and bioinformatics layer, the machine-learning layer, the Python backend, and the web frontend. An optional GPT-based layer can turn structured results into an easier-to-read explanation.
The Whole Project in Simple Words
 Someone using the completed system does not need to understand machine learning, Python, single-cell analysis or bioinformatics. They provide or select the two targets they want to analyze. The application finds the information needed for those targets, converts that information into numerical features, sends the features to the trained model, receives the model result, and displays the result in the website.
 


What Problem Does It Solve? 
There can be many possible combinations of biological targets. Evaluating every possible combination manually can be difficult because each target can have information about expression, cell types, tissues, pathways, safety-related properties and clinical development. The project provides a computational way to organize this information and compare target pairs using a consistent machine-learning process. The result is intended as research decision support, not as clinical proof.
Project Input and Output
The project uses more than one kind of input. The most important input is the identity of the two targets being analysed. Behind those target names is a larger collection of biological and engineered information.
At the application level, the input can be as simple as selecting two targets, for example Target A + Target B. The user does not manually enter hundreds of biological measurements. The system uses the stored and processed information associated with those targets to construct the model input.
Data Preprocessing
Here we the data is manipulated, the given dataset is cleaned- removes unwanted data, outliers
Feature Engineering — Turning Biology into Numbers
Feature engineering is the process of converting useful biological information into numerical measurements that a machine-learning algorithm can use.
For BI Spec Pairwise AI, this is particularly important because the system needs to describe a pair, not just two isolated target names.
Machine Learning
The machine-learning stage uses the processed feature table to learn patterns associated with the project's target-pair objective. XGBoost is used as the main tree-based learning method for structured/tabular features, together with a pairwise or ranking-oriented learning setup where appropriate.
XGBoost is well suited to structured data containing many numerical features and interactions between features. It can model nonlinear relationships and is commonly used for tabular machine-learning problems. The model does not receive raw biological text or a webpage. It receives the engineered numerical representation of the target pair.
Frontend
 The frontend is the part of the project that the user sees in a web browser. It was designed to hide the complexity of the data and machine-learning pipeline behind a simple interface. The technology used are HTML, JAVASCRIPT, CSS, HTTP.

Backend
 • Receive the target-pair request from the frontend. 
• Check that the request contains valid information.
 • Find the required target and feature data. 
• Prepare the feature representation required by the model. 
• Load and use the trained machine-learning model. 
• Generate the prediction or ranking result. 
• Prepare feature-importance or SHAP information. 
• Return the result to the frontend in a structured format. 
• Optionally request a GPT-generated explanation using the structured result.

Final System Architecture 
BIOLOGICAL / CLINICAL DATA
↓
PREPROCESSING
↓
SCANPY + FEATURE ENGINEERING
↓
PAIRWISE ML DATA
↓
XGBOOST / RANKING MODEL
↓
MODEL EVALUATION
↓
SAVED TRAINED MODEL
↓
FASTAPI BACKEND
↕
HTML + CSS + JAVASCRIPT FRONTEND BI Spec Pairwise AI | Completed Project Explanation
↓
SHAP / OPTIONAL GPT EXPLANATION
↓
FINAL TARGET-PAIR ANALYSIS

