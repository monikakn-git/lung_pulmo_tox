import shap
import joblib
import pandas as pd

EXPLAINER = None

def get_explainer():
    global EXPLAINER
    if EXPLAINER is None:
        model = joblib.load("model.joblib")
        reference_data = joblib.load("reference_data.joblib")
        # We use a summary of background data to keep it fast
        background = shap.sample(reference_data['X_train'], 50)
        EXPLAINER = shap.TreeExplainer(model, background, feature_perturbation='interventional')
    return EXPLAINER

def explain_prediction(features_df, risk_category):
    """
    Generate SHAP explanation text and top features for the input.
    """
    explainer = get_explainer()
    shap_values = explainer(features_df)
    
    # Extract feature importance for the single prediction
    values = shap_values.values[0]
    feature_names = features_df.columns
    
    # Create a dataframe of feature contributions
    contributions = pd.DataFrame({
        'feature': feature_names,
        'contribution': values
    })
    
    contributions['abs_contribution'] = contributions['contribution'].abs()
    contributions = contributions.sort_values(by='abs_contribution', ascending=False)
    
    top_features = contributions.head(5)
    
    # Generate simple textual explanation
    if risk_category == "HIGH RISK" or risk_category == "MEDIUM RISK":
        explanation_text = (
            "This molecule contains structural properties strongly associated with known lung-toxic compounds. "
        )
        driving_feats = top_features[top_features['contribution'] > 0]
        if not driving_feats.empty:
            explanation_text += f"Specifically, features like {', '.join(driving_feats['feature'].tolist()[:3])} increase the risk."
    else:
        explanation_text = (
            "This molecule lacks the typical structural patterns associated with pulmonary toxicity. "
        )
        protective_feats = top_features[top_features['contribution'] < 0]
        if not protective_feats.empty:
            explanation_text += f"Properties such as {', '.join(protective_feats['feature'].tolist()[:3])} reduce the risk."

    return explanation_text, shap_values
