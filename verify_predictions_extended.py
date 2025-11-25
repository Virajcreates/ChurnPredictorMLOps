import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Initialize Rich Console for pretty printing
console = Console()

# Load model
PIPELINE_PATH = Path("churn_pipeline.pkl")
if not PIPELINE_PATH.exists():
    console.print("[bold red]❌ Model file not found![/bold red]")
    exit()

console.print("[bold blue]Loading model...[/bold blue]")
pipeline = joblib.load(PIPELINE_PATH)
console.print("[bold green]✅ Model loaded.[/bold green]\n")

# Define 10 diverse test cases
test_cases = [
    {
        "name": "1. 🔴 High Risk: New, Month-to-Month, Fiber",
        "desc": "Classic churner profile: New customer, no commitment, expensive service.",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.00, "TotalCharges": 70.00
        }
    },
    {
        "name": "2. 🟢 Low Risk: Loyal, 2-Year, DSL",
        "desc": "Ideal customer: Long tenure, committed contract, automatic payments.",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 72, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 85.00, "TotalCharges": 6120.00
        }
    },
    {
        "name": "3. 🟡 Borderline: Mid-Tenure, Month-to-Month",
        "desc": "Unsure: Has stayed a while (1 year) but still on risky contract.",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 90.00, "TotalCharges": 1080.00
        }
    },
    {
        "name": "4. 🟢 Low Risk: Senior, Long Tenure, Basic",
        "desc": "Stable senior: Long history, basic service, reliable.",
        "data": {
            "gender": "Female", "SeniorCitizen": 1, "Partner": "Yes", "Dependents": "No",
            "tenure": 50, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "Yes", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "One year",
            "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 60.00, "TotalCharges": 3000.00
        }
    },
    {
        "name": "5. 🔴 High Risk: Senior, New, Fiber",
        "desc": "Risky senior: New, high tech needs but no support services.",
        "data": {
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
            "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.00, "TotalCharges": 285.00
        }
    },
    {
        "name": "6. 🟢 Low Risk: No Internet (Phone Only)",
        "desc": "Simple user: Phone only, very low churn risk usually.",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "No", "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service", "TechSupport": "No internet service", "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service", "Contract": "One year",
            "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 20.00, "TotalCharges": 480.00
        }
    },
    {
        "name": "7. 🟡 Mixed: Family Plan, Fiber, Month-to-Month",
        "desc": "Conflicting signals: Family ties (good) but risky contract/service.",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 20, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 105.00, "TotalCharges": 2100.00
        }
    },
    {
        "name": "8. 🔴 High Risk: Tech Savvy, No Support",
        "desc": "Frustrated user? High usage, fiber, but no tech support.",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 8, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 100.00, "TotalCharges": 800.00
        }
    },
    {
        "name": "9. 🟢 Low Risk: New but Committed",
        "desc": "New customer but signed 2-year deal immediately.",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "Yes", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 55.00, "TotalCharges": 110.00
        }
    },
    {
        "name": "10. 🟡 Mixed: Long Tenure, High Cost, Month-to-Month",
        "desc": "Lazy loyalist? Been here long time, but on flexible expensive plan.",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 115.00, "TotalCharges": 6900.00
        }
    }
]

# Load optimal threshold
threshold_path = Path(__file__).parent / "model_threshold.json"
optimal_threshold = 0.5
if threshold_path.exists():
    try:
        with open(threshold_path, "r") as f:
            config = json.load(f)
            optimal_threshold = config.get("threshold", 0.5)
            print(f"✅ Loaded Optimal Threshold: {optimal_threshold:.4f}")
    except Exception:
        pass

    print("\n🔍 Model Verification Results (Ensemble)")
    print("-" * 100)
    print(f"{'Case':<40} | {'Prediction':<10} | {'Prob':<8} | {'Assessment'}")
    print("-" * 100)

    for case in test_cases:
        df = pd.DataFrame([case['data']])
        proba = pipeline.predict_proba(df)[0][1]
        prediction = "Yes" if proba >= optimal_threshold else "No"
    
    # Simple assessment logic
    if "High Risk" in case['name'] and prediction == "Yes":
        assessment = "✅ Correct"
    elif "Low Risk" in case['name'] and prediction == "No":
        assessment = "✅ Correct"
    elif "Borderline" in case['name'] or "Mixed" in case['name']:
        assessment = "🤔 Plausible"
    else:
        assessment = "❓ Review"

    print(f"{case['name'][:40]:<40} : {prediction} ({proba:.1%}) - {assessment}")

print("-" * 100)
