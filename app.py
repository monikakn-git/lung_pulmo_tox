import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from predict import predict_toxicity, load_artifacts
from explain import explain_prediction

# Configure page
st.set_page_config(page_title="Pulmonary Toxicity Predictor", page_icon="🫁", layout="centered")

# Load dataset for drug lookup
try:
    df = pd.read_csv("data.csv")
except Exception as e:
    st.error("Failed to load dataset (data.csv).")
    st.stop()

def lookup_drug(drug_name):
    match = df[df['drug_name'].str.lower() == drug_name.lower()]
    if not match.empty:
        return match.iloc[0]['smiles'], match.iloc[0]['drug_name']
    return None, None

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "About"])

if page == "Home":
    st.title("🫁 Welcome to ToxPredict")
    st.markdown("""
    **Advanced Computational Toxicology at your fingertips.**
    
    In the modern era of pharmaceutical development, identifying adverse effects early in the drug discovery pipeline is paramount. **ToxPredict** leverages state-of-the-art machine learning algorithms to evaluate the pulmonary toxicity of chemical compounds instantaneously.
    
    ### Key Features
    - ⚡ **Instant Analysis:** Evaluate compounds in milliseconds using optimized XGBoost trees.
    - 🔬 **Molecular Precision:** Calculates exact Morgan Fingerprints and structural heuristics.
    - 📊 **Explainable AI:** Understand the 'why' behind every prediction with SHAP values.
    
    Designed for researchers, computational chemists, and healthcare professionals, our platform bridges the gap between complex biochemical interactions and actionable safety insights.
    """)
    st.info("👈 Navigate to the **Dashboard** to analyze a drug.")

elif page == "Dashboard":
    st.title("🫁 Toxicity Dashboard")
    st.markdown("Enter a drug name below to analyze its pulmonary toxicity risk.")
    
    # Load model artifacts
    try:
        if not load_artifacts():
            st.warning("⚠️ Model not trained. Please run `python model.py` to train the model.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
    # Input section
    st.subheader("Analyze Drug")
    drug_input = st.text_input("Enter Drug Name:", placeholder="e.g. Aspirin, Gefitinib")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        predict_btn = st.button("Predict Toxicity", type="primary", use_container_width=True)
    with col2:
        compare_btn = st.button("Compare Mode", use_container_width=True)

    if compare_btn:
        st.session_state.compare_mode = not st.session_state.get('compare_mode', False)

    if st.session_state.get('compare_mode', False):
        drug_input_2 = st.text_input("Enter Second Drug Name (for comparison):", placeholder="e.g. Loratadine")

    def display_results(drug_name, title="Results"):
        st.subheader(title)
        
        smiles, actual_drug_name = lookup_drug(drug_name)
        if not smiles:
            st.error(f"❌ Drug '{drug_name}' not found in database. Please try another name.")
            return

        try:
            with st.spinner(f"Analyzing {actual_drug_name}..."):
                result = predict_toxicity(smiles)
                explanation_text, shap_values = explain_prediction(result['features'], result['risk_category'])
            
            # Display main metrics
            rc_col, prob_col, conf_col = st.columns(3)
            
            # Color coding
            risk = result['risk_category']
            color = "red" if risk == "HIGH RISK" else "orange" if risk == "MEDIUM RISK" else "green"
            
            with rc_col:
                st.markdown(f"**Risk Category**")
                st.markdown(f"<h3 style='color: {color};'>{risk}</h3>", unsafe_allow_html=True)
                
            with prob_col:
                st.metric("Toxicity Probability", f"{result['probability']:.1%}")
                
            with conf_col:
                conf = result['confidence_score']
                st.metric("Confidence Score", f"{conf:.1f}%", help="Based on molecular similarity to training data")

            # Explainability
            st.markdown("---")
            st.markdown("**AI Explanation:**")
            st.info(explanation_text)
            
            with st.expander("Show Detailed Feature Importance (SHAP)"):
                st.markdown("This plot shows the features (Morgan fingerprint bits & properties) that drove the prediction.")
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig, bbox_inches='tight')
                
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

    # Trigger prediction
    if predict_btn and drug_input:
        display_results(drug_input, title=f"Results for {drug_input}")
        
        if st.session_state.get('compare_mode', False) and 'drug_input_2' in locals() and drug_input_2:
            st.markdown("---")
            display_results(drug_input_2, title=f"Comparison Results for {drug_input_2}")
            
    elif predict_btn:
        st.warning("Please enter a valid drug name.")

elif page == "About":
    st.title("📖 About the Science")
    st.markdown("""
    ### The Challenge of Pulmonary Toxicity
    Drug-induced pulmonary toxicity is a severe, sometimes fatal, complication of certain medications (e.g., chemotherapeutics like Bleomycin or anti-arrhythmics like Amiodarone). Because the lung is highly vascularized and exposed to high concentrations of oxygen, it is uniquely susceptible to oxidative stress and toxic accumulation. Identifying these risks traditionally required costly and time-consuming in vivo trials.

    ### Our Methodology
    ToxPredict shifts the paradigm by utilizing an **XGBoost machine learning architecture** trained on a heavily curated dataset of FDA-approved drugs. By converting the chemical structure of a drug into a mathematical representation known as a **Morgan Fingerprint**, the model can "see" the structural motifs that historically correlate with lung damage.

    ### Key Physicochemical Descriptors
    Alongside structural fingerprints, our algorithm calculates key thermodynamic and spatial properties of the molecule:
    - **Molecular Weight:** Influences tissue penetration and accumulation rates.
    - **LogP (Lipophilicity):** High lipophilicity often correlates with deeper cellular membrane permeation.
    - **Topological Polar Surface Area (TPSA):** Helps predict transport across biological barriers.
    - **Number of Rotatable Bonds:** A measure of molecular flexibility.
    - **Hydrogen Bond Donors/Acceptors:** Crucial for understanding target binding affinity.

    ### Technology Stack & Explainability
    The backend is powered by **Python, FastAPI, and RDKit** for high-performance chemoinformatics. The predictive engine relies on **scikit-learn and XGBoost**. To avoid the "black box" problem of AI, we integrated **SHAP (SHapley Additive exPlanations)**, ensuring that every risk assessment comes with a transparent breakdown of exactly which molecular features drove the algorithm's decision.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Developed for Hackathon | Powered by XGBoost & RDKit")
