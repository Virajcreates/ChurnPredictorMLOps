import joblib
import shap
import xgboost as xgb
import json

try:
    pipeline = joblib.load("churn_pipeline.pkl")
    classifier = pipeline.named_steps['model']
    print("Model loaded.")
    
    booster = classifier.get_booster()
    config = json.loads(booster.save_config())
    print("Base score from config:", config.get('learner', {}).get('learner_model_param', {}).get('base_score'))
    
    print("Attempting to create TreeExplainer...")
    explainer = shap.TreeExplainer(booster)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
