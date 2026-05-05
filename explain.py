import shap
import joblib
import pandas as pd
import numpy as np

EXPLAINER = None
IS_TREE = True

def get_explainer():
    global EXPLAINER, IS_TREE
    if EXPLAINER is None:
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(base_dir, "model.joblib"))
        reference_data = joblib.load(os.path.join(base_dir, "reference_data.joblib"))
        
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
        elif len(values.shape) == 2: # (samples, features)
            values = values[0]
        else:
            values = values.ravel()
    else:
        # KernelExplainer returns a list or array
        shap_values = explainer.shap_values(features_df)
        if isinstance(shap_values, list):
            # For binary classification with list output, class 1 is usually at index 1
            # If length is 1, it's already the positive class or just one class
            v = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            values = v[0] if len(v.shape) > 1 else v
        else:
            if len(shap_values.shape) == 3:
                values = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:
                values = shap_values[0]
            else:
                values = shap_values.ravel()
        
    # Final safety check: ensure values is 1D and matches feature_names length
    values = np.array(values).flatten().tolist()
    feature_names = features_df.columns.tolist()
    
    if len(values) != len(feature_names):
        # Fallback if there's a mismatch
        values = [0.0] * len(feature_names)

    try:
        # Create a dataframe of feature contributions using list of tuples for maximum stability
        data_list = list(zip(feature_names, values))
        contributions = pd.DataFrame(data_list, columns=['feature', 'contribution'])
    except Exception as e:
        # Fallback explanation if pandas still fails
        return f"Explanation generated an error: {str(e)}", np.zeros(len(feature_names))
    
    contributions['abs_contribution'] = contributions['contribution'].abs()
    contributions = contributions.sort_values(by='abs_contribution', ascending=False)
    
    top_features = contributions.head(10)
    top_features_list = []
    for _, row in top_features.iterrows():
        top_features_list.append({
            "feature": row['feature'],
            "value": float(row['contribution'])
        })
    
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

    return explanation_text, top_features_list
