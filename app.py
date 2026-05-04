import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from predict import predict_toxicity, load_artifacts
from explain import explain_prediction

# Configure page
st.set_page_config(page_title="Pulmonary Toxicity Predictor", page_icon="🫁", layout="centered")

st.title("🫁 Pulmonary Toxicity Predictor")
st.markdown("Predict whether a drug is likely to cause lung toxicity based on its molecular structure (SMILES).")

# Load model artifacts
try:
    if not load_artifacts():
        st.warning("⚠️ Model not trained. Please run `python model.py` to train the model.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input section
st.subheader("Analyze Molecule")
smiles_input = st.text_input("Enter SMILES String:", placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O")

col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button("Predict Toxicity", type="primary", use_container_width=True)

with col2:
    compare_btn = st.button("Compare Mode", use_container_width=True)

if compare_btn:
    st.session_state.compare_mode = not st.session_state.get('compare_mode', False)

if st.session_state.get('compare_mode', False):
    smiles_input_2 = st.text_input("Enter Second SMILES String (for comparison):", placeholder="e.g. C1=CC=C(C=C1)N")

def display_results(smiles, title="Results"):
    st.subheader(title)
    try:
        with st.spinner("Analyzing structure..."):
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
            # Confidence based on Tanimoto similarity to training set
            conf = result['confidence_score']
            st.metric("Confidence Score", f"{conf:.1f}%", help="Based on molecular similarity to training data")

        # Explainability
        st.markdown("---")
        st.markdown("**AI Explanation:**")
        st.info(explanation_text)
        
        with st.expander("Show Detailed Feature Importance (SHAP)"):
            st.markdown("This plot shows the features (Morgan fingerprint bits & properties) that drove the prediction.")
            # We use matplotlib to render the shap plot
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig, bbox_inches='tight')
            
    except ValueError as ve:
        st.error(f"❌ Invalid SMILES string provided: {smiles}")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

# Trigger prediction
if predict_btn and smiles_input:
    display_results(smiles_input)
    
    if st.session_state.get('compare_mode', False) and 'smiles_input_2' in locals() and smiles_input_2:
        st.markdown("---")
        display_results(smiles_input_2, title="Comparison Results")
        
elif predict_btn:
    st.warning("Please enter a valid SMILES string.")

st.markdown("---")
st.caption("Developed for Hackathon | Powered by XGBoost & RDKit")
