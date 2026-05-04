import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pubchempy as pcp
from predict import predict_toxicity, load_artifacts
from explain import explain_prediction
from rdkit import Chem
from rdkit.Chem import Draw
import os

# Configure page
st.set_page_config(page_title="Pulmonary Toxicity Predictor", page_icon="🫁", layout="centered")

# Load dataset for drug lookup
try:
    df = pd.read_csv("data.csv")
except Exception as e:
    st.error("Failed to load dataset (data.csv).")
    st.stop()

def lookup_drug(drug_name):
    # Try local database first
    match = df[df['drug_name'].str.lower() == drug_name.lower()]
    if not match.empty:
        return match.iloc[0]['smiles'], match.iloc[0]['drug_name']
    
    # Try PubChem as fallback
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            return compounds[0].canonical_smiles, drug_name
    except Exception:
        pass
    return None, None

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Model Performance", "About"])

if page == "Home":
    st.title("🫁 Welcome to ToxPredict")
    
    with st.container():
        st.markdown("### Advanced Computational Toxicology")
        st.write("In the modern era of pharmaceutical development, identifying adverse effects early in the drug discovery pipeline is paramount. **ToxPredict** leverages state-of-the-art machine learning algorithms to evaluate the pulmonary toxicity of chemical compounds instantaneously.")

    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("⚡ **Instant Analysis**\n\nEvaluate compounds in milliseconds using optimized XGBoost trees.")
    with col2:
        st.success("🔬 **Molecular Precision**\n\nCalculates exact Morgan Fingerprints and structural heuristics.")
    with col3:
        st.warning("📊 **Explainable AI**\n\nUnderstand the 'why' behind every prediction with SHAP values.")

    st.markdown("---")
    
    st.markdown("### Our Analysis Pipeline")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Step 1", "Data Input", delta="Drug Name")
    p_col2.metric("Step 2", "Extraction", delta="Features")
    p_col3.metric("Step 3", "Inference", delta="XGBoost")
    p_col4.metric("Step 4", "Explain", delta="SHAP")

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
    drug_input = st.text_input("Enter Drug Name (e.g. Aspirin) or SMILES:", placeholder="e.g. Aspirin, Gefitinib, CC(=O)OC1=CC=CC=C1C(=O)O")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        predict_btn = st.button("Predict Toxicity", type="primary", use_container_width=True)
    with col2:
        compare_btn = st.button("Compare Mode", use_container_width=True)

    if compare_btn:
        st.session_state.compare_mode = not st.session_state.get('compare_mode', False)

    if st.session_state.get('compare_mode', False):
        drug_input_2 = st.text_input("Enter Second Drug Name or SMILES (for comparison):", placeholder="e.g. Loratadine")

    def display_results(input_text, title="Results"):
        st.subheader(title)
        
        # Check if input is SMILES or Name
        mol = Chem.MolFromSmiles(input_text)
        if mol:
            smiles = input_text
            actual_drug_name = "Custom Molecule"
        else:
            smiles, actual_drug_name = lookup_drug(input_text)
            if not smiles:
                st.error(f"❌ Drug '{input_text}' not found in database or PubChem. Please try a different name or provide a SMILES string.")
                return

        try:
            with st.spinner(f"Analyzing {actual_drug_name}..."):
                result = predict_toxicity(smiles)
                explanation_text, shap_values = explain_prediction(result['features'], result['risk_category'])
            
            # Result layout: Image and Metrics
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("**Molecular Structure**")
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(500, 500))
                    st.image(img, use_container_width=True)
                else:
                    st.warning("Structure not available")

            with res_col2:
                # Color coding
                risk = result['risk_category']
                color = "red" if risk == "HIGH RISK" else "orange" if risk == "MEDIUM RISK" else "green"
                
                st.markdown(f"**Risk Category**")
                st.markdown(f"<h3 style='color: {color};'>{risk}</h3>", unsafe_allow_html=True)
                
                st.metric("Toxicity Probability", f"{result['probability']:.1%}")
                
                conf = result['confidence_score']
                st.metric("Confidence Score", f"{conf:.1f}%", help="Based on molecular similarity to training data")

            # Explainability
            st.markdown("---")
            st.markdown("**AI Explanation:**")
            st.info(explanation_text)
            
            with st.expander("Show Detailed Feature Importance"):
                st.markdown("This plot shows the top 10 features that drove this specific prediction.")
                
                # Get top features from the contributions
                feature_names = result['features'].columns
                contrib_df = pd.DataFrame({'feature': feature_names, 'val': shap_values})
                contrib_df['abs_val'] = contrib_df['val'].abs()
                top_10 = contrib_df.sort_values('abs_val', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#ff2a5f' if x > 0 else '#00ffa2' for x in top_10['val']]
                ax.barh(top_10['feature'], top_10['val'], color=colors)
                ax.set_xlabel('SHAP Value (Contribution)')
                ax.set_title('Top 10 Feature Contributions')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"⚠️ An error occurred during analysis: {e}")

    # Trigger prediction
    if predict_btn and drug_input:
        display_results(drug_input, title=f"Results for {drug_input}")
        
        if st.session_state.get('compare_mode', False) and 'drug_input_2' in locals() and drug_input_2:
            st.markdown("---")
            display_results(drug_input_2, title=f"Comparison Results for {drug_input_2}")
            
    elif predict_btn:
        st.warning("Please enter a drug name or SMILES string.")

elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown("Analysis of the model's predictive accuracy.")
    
    if os.path.exists("frontend/model_comparison.png"):
        st.image("frontend/model_comparison.png", use_container_width=True)
    else:
        st.warning("Performance metrics are being updated.")

elif page == "About":
    st.title("📖 About ToxPredict")
    st.write("ToxPredict is an AI-driven tool for identifying pulmonary toxicity in drug candidates.")

st.sidebar.markdown("---")
st.sidebar.caption("Developed for Hackathon | Powered by XGBoost & RDKit")
