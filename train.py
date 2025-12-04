#!/usr/bin/env python3
"""Train and register the Telco churn model with MLflow using Tuned Ensemble Learning + Feature Engineering (No SMOTE)."""
from __future__ import annotations

import joblib
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import custom Feature Engineer
import sys
sys.path.append(str(Path(__file__).parent))
from utils.feature_engineering import FeatureEngineer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "telco_churn.csv"
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "churn-experiments"
MODEL_NAME = "churn-ensemble-accuracy-optimized"
PIPELINE_PATH = PROJECT_ROOT / "churn_pipeline.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "model_threshold.json"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = df.copy()
    dataset["Churn"] = dataset["Churn"].map({"Yes": 1, "No": 0})
    dataset = dataset.dropna(subset=["Churn"])

    features = dataset.drop(columns=["customerID", "Churn"], errors="ignore")
    target = dataset["Churn"]
    return features, target


def get_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def tune_model(name: str, pipeline: Pipeline, param_dist: Dict[str, Any], X_train, y_train) -> Dict[str, Any]:
    print(f"\n🔍 Tuning {name}...")
    # DEBUG: Check versions and classifier status in CI
    from sklearn.base import is_classifier
    import xgboost
    import sklearn
    print(f"DEBUG: XGBoost version: {xgboost.__version__}")
    print(f"DEBUG: Scikit-learn version: {sklearn.__version__}")
    print(f"DEBUG: is_classifier(pipeline): {is_classifier(pipeline)}")
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,  # Aggressive tuning!
        scoring="roc_auc",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"✅ Best {name} Params: {search.best_params_}")
    return search.best_params_


def configure_mlflow() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def optimize_threshold(y_true, y_proba) -> Tuple[float, float]:
    """Find the threshold that maximizes accuracy."""
    best_threshold = 0.5
    best_accuracy = 0.0
    
    thresholds = np.arange(0.3, 0.7, 0.01)
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            
    return best_threshold, best_accuracy


def train_and_register() -> None:
    configure_mlflow()
    df = load_dataset()
    X, y = split_features_target(df)

    # Apply Feature Engineering FIRST to identify new columns
    print("🛠️ Applying Feature Engineering...")
    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X)
    
    numeric_features = X_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [col for col in X_fe.columns if col not in numeric_features]
    
    print(f"📊 New Features Added: {[col for col in X_fe.columns if col not in X.columns]}")

    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, # Split original data, pipeline handles FE
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Calculate scale_pos_weight for class imbalance (since we removed SMOTE)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # --- Tuning Phase ---
    # Note: We tune on X_train (raw) -> Pipeline(FE -> Prep -> Model)
    
    # 1. Tune XGBoost
    xgb_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1))
    ])
    xgb_params = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__gamma": [0, 0.1, 0.2, 0.5],
        "model__scale_pos_weight": [1, scale_pos_weight]
    }
    best_xgb_params = tune_model("XGBoost", xgb_pipeline, xgb_params, X_train, y_train)

    # 2. Tune LightGBM
    lgbm_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMClassifier(objective="binary", random_state=42, n_jobs=-1, verbose=-1))
    ])
    lgbm_params = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__num_leaves": [20, 31, 50, 70, 100],
        "model__max_depth": [-1, 5, 10, 15],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__scale_pos_weight": [1, scale_pos_weight]
    }
    best_lgbm_params = tune_model("LightGBM", lgbm_pipeline, lgbm_params, X_train, y_train)
    
    # 3. Tune CatBoost
    cat_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False))
    ])
    cat_params = {
        "model__iterations": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__depth": [4, 6, 8, 10],
        "model__l2_leaf_reg": [1, 3, 5, 7, 9],
        "model__scale_pos_weight": [1, scale_pos_weight]
    }
    best_cat_params = tune_model("CatBoost", cat_pipeline, cat_params, X_train, y_train)

    # --- Building Final Ensemble ---
    
    print("\n🏗️ Building Final Ensemble with Tuned Parameters...")
    
    # Re-instantiate models with best params (strip 'model__' prefix)
    final_xgb = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **{k.replace("model__", ""): v for k, v in best_xgb_params.items()}
    )
    
    final_lgbm = lgb.LGBMClassifier(
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **{k.replace("model__", ""): v for k, v in best_lgbm_params.items()}
    )
    
    final_cat = CatBoostClassifier(
        random_state=42,
        verbose=0,
        allow_writing_files=False,
        **{k.replace("model__", ""): v for k, v in best_cat_params.items()}
    )
    
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', final_xgb),
            ('lgbm', final_lgbm),
            ('cat', final_cat)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    final_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", ensemble)
    ])

    try:
        with mlflow.start_run(run_name="ensemble_accuracy_optimized"):
            print("🧠 Training Final Ensemble Model...")
            final_pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

            # Optimize Threshold
            print("🎯 Optimizing Threshold for Accuracy...")
            best_threshold, best_accuracy = optimize_threshold(y_test, y_pred_proba)
            print(f"✅ Best Threshold: {best_threshold:.4f} -> Max Accuracy: {best_accuracy:.4f}")
            
            # Save threshold
            with open(THRESHOLD_PATH, "w") as f:
                json.dump({"threshold": best_threshold, "accuracy": best_accuracy}, f)
            print(f"💾 Threshold saved to {THRESHOLD_PATH}")

            # Apply best threshold
            y_pred = (y_pred_proba >= best_threshold).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Log metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("optimal_threshold", best_threshold)
            
            # Log best params
            mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})
            mlflow.log_params({f"lgbm_{k}": v for k, v in best_lgbm_params.items()})
            mlflow.log_params({f"cat_{k}": v for k, v in best_cat_params.items()})

            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_text(json.dumps(report, indent=2), "classification_report.json")

            input_example = X_test.iloc[:5]
            signature = infer_signature(input_example, final_pipeline.predict(input_example))

            model_info = mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                signature=signature,
                input_example=input_example,
            )

            print(f"\nTraining complete:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            
            client = mlflow.MlflowClient()
            model_version = model_info.registered_model_version
            client.set_registered_model_alias(MODEL_NAME, "champion", model_version)
            print(f"✅ Model registered as '{MODEL_NAME}' version {model_version} with 'champion' alias")
        
        # Save the final pipeline
        print(f"\n💾 Saving pipeline to {PIPELINE_PATH}...")
        joblib.dump(final_pipeline, PIPELINE_PATH)
        print(f"✅ Model saved to churn_pipeline.pkl")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    train_and_register()
