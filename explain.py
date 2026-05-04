import shap
import joblib
import pandas as pd
import numpy as np

EXPLAINER = None
IS_TREE = True

def get_explainer():
    global EXPLAINER, IS_TREE
    if EXPLAINER is None:
        model = joblib.load("model.joblib")
        reference_data = joblib.load("reference_data.joblib")
        
        # Check if model is tree-based
        model_name = reference_data.get("model_name", "")
        IS_TREE = model_name in ["XGBoost", "RandomForest", "ExtraTrees"]
        
        if IS_TREE:
            background = shap.sample(reference_data['X_train'], 50)
            EXPLAINER = shap.TreeExplainer(model, background, feature_perturbation='interventional')
        else:
            # For Artificial Neural Networks or Logistic Regression
            # K-means to summarize the background to keep KernelExplainer fast
            background = shap.kmeans(reference_data['X_train'], 10)
            # KernelExplainer needs a function that predicts probabilities for the positive class
            def predict_func(X):
                return model.predict_proba(X)[:, 1]
            EXPLAINER = shap.KernelExplainer(predict_func, background)
            
    return EXPLAINER

def explain_prediction(features_df, risk_category):
    """
    Generate SHAP explanation text and top features for the input.
    """
    explainer = get_explainer()
    
    if IS_TREE:
        shap_values_obj = explainer(features_df)
        values = shap_values_obj.values
        # Handle cases where SHAP returns values for both classes (e.g., RandomForest)
        if len(values.shape) == 3: # (samples, features, classes)
            values = values[0, :, 1]
        else:
            values = values[0]
    else:
        # KernelExplainer returns a list or array
        shap_values = explainer.shap_values(features_df)
        if isinstance(shap_values, list):
            # For multi-class, pick class 1
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            if len(shap_values.shape) == 3:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]
        
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

    return explanation_text, values
