import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

# Load model
PIPELINE_PATH = Path("churn_pipeline.pkl")
if not PIPELINE_PATH.exists():
    print("❌ Model file not found!")
    exit()

print("Loading model...")
pipeline = joblib.load(PIPELINE_PATH)
print("✅ Model loaded.")

# Define test cases
test_cases = [
    {
        "name": "🔴 High Risk (New, Month-to-Month, Fiber, High Cost)",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.50, "TotalCharges": 191.00
        }
    },
    {
        "name": "🟢 Low Risk (Loyal, 2-Year, DSL, Low Cost)",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 70, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 65.00, "TotalCharges": 4550.00
        }
    }
]

def explain_prediction_ensemble(pipeline, customer_data_df):
    """Simplified SHAP explanation for Ensemble."""
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['model']
    
    transformed_data = preprocessor.transform(customer_data_df)
    feature_names = preprocessor.get_feature_names_out()
    
    agg_shap_values = None
    estimators = classifier.estimators_
    valid_count = 0
    
    for model in estimators:
        try:
            if hasattr(model, 'get_booster'): # XGBoost
                explainer = shap.TreeExplainer(model.get_booster())
            elif hasattr(model, 'booster_'): # LightGBM
                explainer = shap.TreeExplainer(model.booster_)
            else: # CatBoost
                explainer = shap.TreeExplainer(model)
                
            shap_values = explainer.shap_values(transformed_data)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            if len(shap_values.shape) == 3:
                shap_values = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:
                if shap_values.shape[0] == 1:
                    shap_values = shap_values[0]
            
            if agg_shap_values is None:
                agg_shap_values = np.zeros_like(shap_values)
            
            agg_shap_values += shap_values
            valid_count += 1
        except Exception as e:
            continue
            
    if valid_count > 0:
        return agg_shap_values / valid_count, feature_names
    return None, None

print("\n🔍 Verifying Predictions:\n")

for case in test_cases:
    print(f"--- {case['name']} ---")
    df = pd.DataFrame([case['data']])
    
    # Predict
    proba = pipeline.predict_proba(df)[0][1]
    prediction = "Yes" if proba >= 0.5 else "No"
    
    print(f"🔮 Prediction: {prediction}")
    print(f"📊 Probability: {proba:.2%}")
    
    # Explain
    shap_vals, feats = explain_prediction_ensemble(pipeline, df)
    
    if shap_vals is not None:
        # Get top 3 drivers
        indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
        print("💡 Top 3 Drivers:")
        for i in indices:
            impact = "⬆️ Increases Risk" if shap_vals[i] > 0 else "⬇️ Decreases Risk"
            print(f"   - {feats[i]}: {shap_vals[i]:.4f} ({impact})")
    print("\n")
