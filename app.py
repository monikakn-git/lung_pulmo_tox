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
    **Determine the likelihood of pulmonary toxicity directly from a drug's name.**
    
    Our model uses advanced machine learning (XGBoost) combined with molecular structure analysis to predict potential lung toxicity risks. Built for researchers and healthcare professionals to evaluate compounds efficiently.
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
    st.title("📖 About the Project")
    st.markdown("""
    ### Methodology
    This tool utilizes an XGBoost machine learning model trained on a curated dataset of drugs known to cause pulmonary toxicity alongside safe alternatives. It calculates the Morgan Fingerprints from the molecular SMILES strings to identify structural patterns correlated with lung toxicity.

    ### Features Evaluated
    - Molecular Weight
    - LogP (Lipophilicity)
    - Topological Polar Surface Area (TPSA)
    - Number of Rotatable Bonds
    - Hydrogen Bond Donors/Acceptors

    ### Technology Stack
    Built using Python, FastAPI, scikit-learn, XGBoost, Streamlit, and SHAP for explainability.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Developed for Hackathon | Powered by XGBoost & RDKit")
