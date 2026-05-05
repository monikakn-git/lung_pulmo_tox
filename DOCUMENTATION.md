# 📘 ToxPredict — Technical Documentation

> **For an overview of the project, features, and setup instructions, see the [README.md](README.md).**

This document provides in-depth technical details on the scientific methodology, tooling, and system architecture of ToxPredict.

---

## 📑 Table of Contents

- [Methodology](#-methodology)
- [Data Curation & Provenance](#-data-curation--provenance)
- [Feature Engineering Pipeline](#-feature-engineering-pipeline)
- [Model Training & Validation](#-model-training--validation)
- [System Architecture](#-system-architecture)
- [Explainability Framework](#-explainability-framework)
- [Tools & Dependencies](#-tools--dependencies)
- [API Reference](#-api-reference)

---

## 🔬 Methodology

ToxPredict employs a rigorous computational chemistry pipeline to translate molecular structures into toxicological risk assessments. The end-to-end process follows four stages:

1.  **Data Curation** — Aggregation and labeling of pharmaceutical compounds from clinical databases.
2.  **Feature Engineering** — Extraction of structural fingerprints and physicochemical descriptors.
3.  **Preprocessing** — Normalization and scaling for uniform model input.
4.  **Inference & Explanation** — Ensemble prediction followed by SHAP-based feature attribution.

---

## 🗃️ Data Curation & Provenance

- **Volume**: 2,400+ pharmaceutical compounds.
- **Sources**:
    - **[PneumoTox](https://www.pneumotox.com/)** — A curated reference for drug-induced respiratory disease.
    - **FAERS** — FDA Adverse Event Reporting System.
    - **SIDER** — Side Effect Resource database.
- **Labeling**: Binary classification (High Risk / Low Risk) for drug-induced lung injury (DILI), based on clinical consensus across all three sources.

---

## ⚙️ Feature Engineering Pipeline

### Structural Fingerprints
*   **Morgan Fingerprints** (ECFP4 equivalent): 2048-bit circular fingerprints with a radius of 2, capturing local molecular environments and substructure patterns.

### Physicochemical Descriptors
| Descriptor | Relevance |
|---|---|
| **Molecular Weight** | Influences tissue penetration and accumulation rates. |
| **LogP (Lipophilicity)** | High lipophilicity correlates with deeper cellular membrane permeation. |
| **TPSA** | Topological Polar Surface Area — predicts transport across biological barriers. |
| **Rotatable Bonds** | A measure of molecular flexibility. |
| **H-Bond Donors/Acceptors** | Crucial for understanding target binding affinity. |

### Preprocessing
*   All features are normalized using `StandardScaler` to ensure uniform weighting across the ensemble's architectural layers.

---

## 🧠 Model Training & Validation

### Validation Strategy
*   **80/20 Stratified Hold-out Split** — Ensures class distribution is preserved.
*   **5-Fold Cross-Validation** — Performed on the training fold to ensure generalizability and prevent overfitting.

### Models Evaluated
| Model | Role | Key Strength |
|---|---|---|
| Random Forest | Base Learner (Level 0) | Captures non-linear feature interactions. |
| Extra Trees | Base Learner (Level 0) | Adds stochasticity to node splitting for robustness. |
| MLP Neural Network | Base Learner (Level 0) | Learns deep representations of molecular fingerprints. |
| XGBoost | Standalone Comparator | High-performance gradient boosting baseline. |
| **Stacking Ensemble** | **Final Model** | **Combines all base learners via XGBoost meta-classifier.** |

### Class Imbalance Handling
*   `scale_pos_weight` is dynamically calculated based on the training set class ratio.
*   Base learners use `class_weight="balanced"` where supported.

---

## 🏗️ System Architecture

The predictive engine is built on a **Stacked Generalization (Heterogeneous Ensemble)** architecture.

### Level 0: Base Learners
*   **Random Forest**: Bagged decision trees for non-linear interaction capture.
*   **Extra Trees**: Enhanced robustness via randomized split thresholds.
*   **Neural Network (MLP)**: Hidden layers `(128, 64)` with early stopping for complex representation learning.

### Level 1: Meta Learner
*   **XGBoost**: 50 estimators, max depth 3, learning rate 0.05. Acts as the final decision-maker, weighing Level 0 outputs to optimize the final toxicity probability.

### Inference Flow
```
Drug Name / SMILES
       │
       ▼
┌──────────────┐    ┌─────────────────┐
│ Local DB     │ OR │ PubChem Lookup  │
│ (data.csv)   │    │ (pubchempy)     │
└──────┬───────┘    └────────┬────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
       ┌──────────────────┐
       │ RDKit Engine     │
       │ → Morgan FP      │
       │ → Descriptors     │
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ StandardScaler   │
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ Stacking Ensemble│
       │ (predict_proba)  │
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ SHAP Explainer   │
       │ (TreeExplainer)  │
       └────────┬─────────┘
                ▼
       Risk Category + Explanation
```

---

## 🔍 Explainability Framework

Every prediction is passed through a **SHAP (SHapley Additive exPlanations)** explainer:

*   Uses `TreeExplainer` for tree-based models (optimal for XGBoost / RF / ET).
*   Decomposes the final probability score back into the **specific molecular features** that drove the decision.
*   The **top 10 features** are surfaced to the user, color-coded by direction of influence:
    *   🔴 **Positive SHAP value** → Increases toxicity risk
    *   🟢 **Negative SHAP value** → Decreases toxicity risk

This ensures **clinical transparency** — every risk assessment can be traced back to concrete molecular evidence.

---

## 🛠️ Tools & Dependencies

| Category | Tool | Purpose |
|---|---|---|
| **Chemoinformatics** | [RDKit](https://www.rdkit.org/) | Molecular parsing, fingerprinting, descriptor calculation |
| **ML — Ensemble** | Scikit-learn | Random Forest, Extra Trees, MLP, Stacking Classifier |
| **ML — Boosting** | XGBoost | Gradient boosted meta-classifier |
| **Explainability** | SHAP | Feature attribution and model interpretability |
| **API** | FastAPI | High-performance REST API with Pydantic validation |
| **Dashboard** | Streamlit | Internal benchmarking and interactive analysis tool |
| **Data** | Pandas | Data manipulation and preprocessing |
| **Visualization** | Matplotlib | Performance charts and scientific plots |
| **Drug Lookup** | PubChemPy | Fallback chemical compound resolution |
| **Frontend** | HTML5 / CSS3 / JS | Production single-page application |
| **Diagrams** | Mermaid.js | Architectural flow diagrams |
| **Hosting** | Vercel / Render | Production deployment infrastructure |

---

## 🌐 API Reference

### `POST /api/predict`

Predicts pulmonary toxicity for a given drug.

**Request Body:**
```json
{
  "drug_name": "Aspirin"
}
```

**Response:**
```json
{
  "drug_name": "Aspirin",
  "smiles": "CC(=O)Oc1ccccc1C(O)=O",
  "risk_category": "LOW RISK",
  "probability": 0.12,
  "confidence_score": 87.5,
  "explanation": "The molecule shows low lipophilicity...",
  "image_base64": "<base64 encoded PNG>",
  "all_models": {
    "XGBoost": 0.11,
    "RandomForest": 0.14,
    "ExtraTrees": 0.09,
    "NeuralNetwork": 0.15,
    "StackingEnsemble": 0.12
  },
  "shap_values": [...],
  "nearest_neighbor": {
    "name": "Ibuprofen",
    "similarity": 0.72,
    "label": "Low Risk"
  }
}
```

### `GET /api/health`

Returns service health status.

```json
{ "status": "healthy" }
```

---

<p align="center"><em>For project setup and usage instructions, see <a href="README.md">README.md</a>.</em></p>
