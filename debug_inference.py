import joblib
import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import traceback

def test_inference():
    print("1. Loading pipeline...")
    try:
        pipeline = joblib.load("churn_pipeline.pkl")
        print("   Pipeline loaded successfully.")
    except Exception as e:
        print(f"   FAILED to load pipeline: {e}")
        return

    print("\n2. Inspecting model...")
    try:
        classifier = pipeline.named_steps['model']
        print(f"   Model type: {type(classifier)}")
        if hasattr(classifier, 'get_booster'):
            booster = classifier.get_booster()
            print("   Booster retrieved.")
            # config = json.loads(booster.save_config())
            # print(f"   Booster config base_score: {config.get('learner', {}).get('learner_model_param', {}).get('base_score')}")
        else:
            print("   No get_booster method.")
    except Exception as e:
        print(f"   Error inspecting model: {e}")

    print("\n3. Creating dummy data...")
    # Create a dummy row matching the training data structure
    # We need to know the columns. We can get them from the preprocessor if possible, or just use a raw dict that matches the app.
    input_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }
    df = pd.DataFrame([input_data])
    print("   Dummy data created.")

    print("\n4. Running Prediction...")
    try:
        # The pipeline expects a DataFrame
        proba = pipeline.predict_proba(df)[0][1]
        print(f"   Prediction successful. Probability: {proba}")
    except Exception as e:
        print(f"   FAILED prediction: {e}")
        traceback.print_exc()
        return

    print("\n5. Running SHAP Explanation...")
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['model']
        
        transformed_data = preprocessor.transform(df)
        print("   Data transformed.")
        
        print("   Initializing TreeExplainer...")
        explainer = shap.TreeExplainer(classifier)
        print("   TreeExplainer initialized.")
        
        feature_names = preprocessor.get_feature_names_out()
        
        print("   Calculating shap_values...")
        shap_values = explainer.shap_values(transformed_data)
        print(f"   SHAP values calculated. Shape: {np.array(shap_values).shape}")
        
    except Exception as e:
        print(f"   FAILED SHAP explanation: {e}")
        traceback.print_exc()
        return

    print("\n6. Testing Plotting...")
    try:
        import matplotlib.pyplot as plt
        
        # Mimic plot_shap_waterfall from app.py
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top 15 features by absolute impact
        indices = np.argsort(np.abs(shap_values))[-15:]
        top_features = [feature_names[i] for i in indices]
        top_values = shap_values[indices]
        
        # Create horizontal bar chart
        colors = ['#ff4b4b' if v > 0 else '#00cc66' for v in top_values]
        ax.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
        ax.set_title('Top 15 Features by Impact', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        print("   Plot created successfully.")
        plt.close(fig)
        
    except Exception as e:
        print(f"   FAILED Plotting: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
