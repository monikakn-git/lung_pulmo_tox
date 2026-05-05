# ToxPredict Technical Documentation

## 🔬 Methodology
ToxPredict employs a rigorous computational chemistry pipeline to translate molecular structures into toxicological risk assessments.

1.  **Data Curation**: 2,400+ pharmaceutical compounds were aggregated from **PneumoTox**, **FAERS**, and **SIDER**. Data was labeled as high/low risk for drug-induced lung injury (DILI) based on clinical consensus.
2.  **Feature Engineering**:
    *   **Structural Fingerprints**: 2048-bit Morgan Fingerprints (ECFP4 equivalent) to capture local molecular environments.
    *   **Physicochemical Descriptors**: Calculation of Lipophilicity (LogP), Molecular Weight, and Topological Polar Surface Area (TPSA) to capture pharmacokinetic trends.
3.  **Preprocessing**: Features are normalized using `StandardScaler` to ensure uniform weighting across architectural layers.
4.  **Validation Strategy**: 80/20 Stratified Hold-out split combined with 5-Fold Cross-Validation on the training fold to ensure generalizability and prevent overfitting.

---

## 🛠️ Tools Used
*   **Chemoinformatics Engine**: [RDKit](https://www.rdkit.org/) for molecular parsing and descriptor calculation.
*   **Machine Learning**:
    *   `Scikit-learn`: Random Forest, Extra Trees, MLP Neural Networks.
    *   `XGBoost`: Gradient Boosted Meta-Classifier.
    *   `SHAP`: SHapley Additive exPlanations for interpretability.
*   **Frontend Stack**:
    *   **Structure**: Semantic HTML5
    *   **Aesthetics**: Vanilla CSS3 (Custom Dark Theme)
    *   **Logic**: Modern JavaScript (ES6+)
    *   **Visualization**: Mermaid.js for architectural diagrams.
*   **Backend Stack**:
    *   **Framework**: FastAPI (High-performance Python API)
    *   **Dashboard**: Streamlit (Internal Benchmarking Tool)
*   **Deployment**:
    *   Vercel (Production Frontend & API)
    *   Render (Benchmark Dashboard)

---

## 🏗️ System Architecture
The predictive engine is built on a **Stacked Generalization (Heterogeneous Ensemble)** architecture, designed to leverage the diverse strengths of different ML algorithms.

### Level 0: Base Learners
*   **Random Forest**: Captures non-linear feature interactions through bagged decision trees.
*   **Extra Trees**: Increases robustness by adding stochasticity to node splitting.
*   **Neural Network (MLP)**: Learns complex deep representations of the molecular fingerprints.

### Level 1: Meta Learner
*   **XGBoost**: A high-performance gradient boosting model that acts as the final decision maker, weighing the outputs of Level 0 models to optimize the final toxicity probability.

### Explainability Layer
*   Every prediction is passed through a **SHAP Explainer**, which decomposes the final score back into the specific molecular features that drove the decision, ensuring clinical transparency.
