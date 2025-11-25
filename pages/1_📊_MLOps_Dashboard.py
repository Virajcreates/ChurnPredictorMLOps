#!/usr/bin/env python3
"""
MLOps Model Comparison Dashboard
Compare all trained models, track performance, and analyze hyperparameters
"""
import streamlit as st
import mlflow
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="MLOps Dashboard", page_icon="📊", layout="wide")

# Clear any Streamlit cache to ensure fresh data
st.cache_data.clear()
st.cache_resource.clear()

# MLflow setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
client = MlflowClient()

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .winner-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 MLOps Model Comparison Dashboard")
st.markdown("**Compare all trained models and select the champion**")
st.markdown("---")

# Show registered models first
st.subheader("🏆 Registered Models")
try:
    registered_models = client.search_registered_models()
    if registered_models:
        for model in registered_models:
            with st.expander(f"📦 {model.name}", expanded=True):
                st.write(f"**Description:** {model.description or 'No description'}")
                
                # Get model versions
                versions = client.search_model_versions(f"name='{model.name}'")
                st.write(f"**Total Versions:** {len(versions)}")
                
                # Show versions in a table
                version_data = []
                for version in versions:
                    aliases = version.aliases if hasattr(version, 'aliases') else []
                    version_data.append({
                        "Version": version.version,
                        "Status": version.status,
                        "Aliases": ", ".join(aliases) if aliases else "None",
                        "Created": version.creation_timestamp
                    })
                
                if version_data:
                    st.dataframe(pd.DataFrame(version_data), use_container_width=True)
    else:
        st.info("No registered models found. Models will be registered when you run `python train.py`")
except Exception as e:
    st.warning(f"Could not load registered models: {e}")

st.markdown("---")
st.subheader("📈 Training Runs Comparison")

# Get all experiments
try:
    experiments = client.search_experiments()
    experiment_names = [exp.name for exp in experiments if exp.name != "Default"]
    
    # Debug info
    st.sidebar.info(f"🔍 Debug Info:\n- Total experiments: {len(experiments)}\n- Non-default experiments: {len(experiment_names)}")
    
    if not experiment_names:
        st.info("ℹ️ **MLOps Dashboard** tracks models trained with MLflow experiment tracking.")
        st.markdown("""
        ### 🎯 What is this dashboard?
        
        This dashboard compares **multiple model training runs** side-by-side to help you:
        - Compare different hyperparameters
        - Track model performance over time
        - Select the best "champion" model
        
        ### 📊 Current Status
        
        **Your current model (`churn_pipeline.pkl`) is working perfectly!** ✅
        
        This dashboard is for **advanced MLOps workflows** where you:
        1. Train multiple models with different hyperparameters
        2. Log each training run to MLflow
        3. Compare and select the best model
        
        ### 🚀 To use this dashboard:
        
        Run the training script with MLflow tracking enabled:
        ```bash
        python train.py
        ```
        
        This will create experiment runs that appear here for comparison.
        
        ---
        
        **Note:** Since you have one production model that's working well, this dashboard is optional for now.
        It's useful when you want to experiment with different model configurations or retrain regularly.
        """)
        st.stop()
    
    selected_experiment = st.selectbox("📁 Select Experiment", experiment_names, index=0)
    experiment = client.get_experiment_by_name(selected_experiment)
    
    # Show experiment details
    st.sidebar.success(f"📂 Experiment: {selected_experiment}\n- ID: {experiment.experiment_id if experiment else 'N/A'}")
    
    if experiment:
        # Get all ACTIVE runs (exclude deleted runs)
        try:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",  # Empty filter to get all active runs
                order_by=["start_time DESC"],
                max_results=50,
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY  # Only show active runs, not deleted
            )
            
            st.sidebar.info(f"📊 Found {len(runs)} active run(s)")
            
        except Exception as e:
            st.error(f"Error searching runs: {e}")
            runs = []
        
        if not runs:
            st.warning("⚠️ No model runs found in this experiment.")
            st.info("""
            **To populate this dashboard:**
            1. Run `python train.py` to train and log a model
            2. The model will appear here automatically
            3. Train multiple times with different settings to compare
            """)
            st.stop()
        
        # Create comparison dataframe
        comparison_data = []
        complete_runs = 0
        incomplete_runs = 0
        
        for run in runs:
            metrics = run.data.metrics
            params = run.data.params
            
            # Handle both old and new metric naming conventions
            accuracy = metrics.get('test_accuracy') or metrics.get('accuracy', 0)
            precision = metrics.get('test_precision') or metrics.get('precision', 0)
            recall = metrics.get('test_recall') or metrics.get('recall', 0)
            f1 = metrics.get('test_f1') or metrics.get('f1', 0)
            roc_auc = metrics.get('test_roc_auc') or metrics.get('roc_auc', 0)
            
            # Check if run has all metrics (either all test_* metrics OR all non-prefixed metrics)
            has_new_metrics = all([
                'test_accuracy' in metrics,
                'test_precision' in metrics,
                'test_recall' in metrics,
                'test_f1' in metrics,
                'test_roc_auc' in metrics
            ])
            
            has_old_metrics = all([
                'accuracy' in metrics,
                'precision' in metrics,
                'recall' in metrics,
                'f1' in metrics,
                'roc_auc' in metrics
            ])
            
            has_all_metrics = has_new_metrics or has_old_metrics
            
            if has_all_metrics:
                complete_runs += 1
                data_quality = "✅ Complete"
            else:
                incomplete_runs += 1
                data_quality = "⚠️ Partial"
            
            comparison_data.append({
                "Run ID": run.info.run_id[:8],
                "Date": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                "Data": data_quality,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
                "n_estimators": params.get('n_estimators', 'N/A'),
                "max_depth": params.get('max_depth', 'N/A'),
                "min_samples_split": params.get('min_samples_split', 'N/A'),
                "Status": run.info.status
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Debug: Show what we found
        st.sidebar.markdown("### 🔍 Debug Info")
        st.sidebar.write(f"Total runs retrieved: {len(runs)}")
        st.sidebar.write(f"DataFrame rows: {len(df)}")
        st.sidebar.write(f"Complete runs: {complete_runs}")
        st.sidebar.write(f"Incomplete runs: {incomplete_runs}")
        
        if len(runs) > 0:
            st.sidebar.write("### Latest run metrics:")
            latest_metrics = runs[0].data.metrics
            for k, v in latest_metrics.items():
                st.sidebar.write(f"  - {k}: {v:.4f}")
        
        # Add debugging info with quality metrics
        if incomplete_runs > 0:
            st.warning(f"⚠️ Found {incomplete_runs} old run(s) with only accuracy metric. Run `python cleanup_mlflow.py` to clean them up, or `python train.py` to add more complete runs.")
        
        st.info(f"📊 Total: {len(df)} run(s) | ✅ Complete: {complete_runs} | ⚠️ Partial: {incomplete_runs}")
        
        # Key Metrics Section
        st.subheader("🎯 Key Performance Indicators")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("🏆 Total Models", len(df), help="Total models trained")
        with col2:
            st.metric("🎯 Best Accuracy", f"{df['Accuracy'].max():.1%}", help="Highest accuracy achieved")
        with col3:
            st.metric("📊 Best Precision", f"{df['Precision'].max():.1%}", help="Highest precision for churn class")
        with col4:
            st.metric("🔍 Best Recall", f"{df['Recall'].max():.1%}", help="Highest recall for churn class")
        with col5:
            st.metric("⚡ Best F1", f"{df['F1-Score'].max():.1%}", help="Highest F1-score")
        with col6:
            st.metric("📈 Best ROC-AUC", f"{df['ROC-AUC'].max():.1%}", help="Highest ROC-AUC score")
        
        st.markdown("---")
        
        # Model Comparison Table
        st.subheader("📋 All Trained Models")
        
        if incomplete_runs > 0:
            with st.expander("ℹ️ Why do some runs show 0% for most metrics?", expanded=False):
                st.markdown("""
                **Old runs only logged accuracy** because your original `train.py` only tracked one metric.
                
                After updating `train.py`, new runs log all 5 metrics:
                - ✅ test_accuracy
                - ✅ test_precision  
                - ✅ test_recall
                - ✅ test_f1
                - ✅ test_roc_auc
                
                **Solutions:**
                1. **Keep old runs** - Shows your progress over time (recommended)
                2. **Delete old runs** - Run `python cleanup_mlflow.py`
                3. **Add more complete runs** - Run `python train.py` multiple times
                """)
        
        st.markdown("*Showing detailed metrics for all training runs*")
        
        # Format the dataframe for better display (but keep original df for charts)
        df_display = df.copy()
        
        # Convert metrics to percentages for display ONLY in the table
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if x > 0 else "0.00%")
        
        # Style the dataframe
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Model Comparison CSV",
            data=csv,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Performance Evolution
        st.subheader("📈 Model Performance Evolution Over Time")
        
        # Use the original df with numeric values for plotting
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Accuracy'], 
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Precision'], 
            mode='lines+markers',
            name='Precision',
            line=dict(color='#f093fb', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Recall'], 
            mode='lines+markers',
            name='Recall',
            line=dict(color='#4facfe', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['F1-Score'], 
            mode='lines+markers',
            name='F1-Score',
            line=dict(color='#43e97b', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['ROC-AUC'], 
            mode='lines+markers',
            name='ROC-AUC',
            line=dict(color='#fa709a', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Model Metrics Over Time",
            xaxis_title="Training Date",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),  # Set y-axis from 0 to 1 for better visualization
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Hyperparameter Analysis
        st.subheader("🔧 Hyperparameter Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # n_estimators vs Accuracy
            fig1 = go.Figure()
            
            # Convert to numeric for plotting
            df_numeric = df.copy()
            df_numeric['n_estimators'] = pd.to_numeric(df_numeric['n_estimators'], errors='coerce')
            df_numeric = df_numeric.dropna(subset=['n_estimators'])
            
            if len(df_numeric) > 0:
                fig1.add_trace(go.Scatter(
                    x=df_numeric['n_estimators'], 
                    y=df_numeric['Accuracy'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=df_numeric['Accuracy'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Accuracy"),
                        line=dict(width=2, color='white')
                    ),
                    text=df_numeric['Run ID'],
                    hovertemplate='<b>Run:</b> %{text}<br>' +
                                  '<b>n_estimators:</b> %{x}<br>' +
                                  '<b>Accuracy:</b> %{y:.3f}<br>' +
                                  '<extra></extra>'
                ))
                
                fig1.update_layout(
                    title="Impact of n_estimators on Accuracy",
                    xaxis_title="n_estimators",
                    yaxis_title="Accuracy",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No numeric n_estimators data available")
        
        with col2:
            # max_depth vs Accuracy
            fig2 = go.Figure()
            
            df_numeric2 = df.copy()
            df_numeric2['max_depth'] = pd.to_numeric(df_numeric2['max_depth'], errors='coerce')
            df_numeric2 = df_numeric2.dropna(subset=['max_depth'])
            
            if len(df_numeric2) > 0:
                fig2.add_trace(go.Scatter(
                    x=df_numeric2['max_depth'], 
                    y=df_numeric2['Accuracy'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=df_numeric2['Accuracy'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Accuracy"),
                        line=dict(width=2, color='white')
                    ),
                    text=df_numeric2['Run ID'],
                    hovertemplate='<b>Run:</b> %{text}<br>' +
                                  '<b>max_depth:</b> %{x}<br>' +
                                  '<b>Accuracy:</b> %{y:.3f}<br>' +
                                  '<extra></extra>'
                ))
                
                fig2.update_layout(
                    title="Impact of max_depth on Accuracy",
                    xaxis_title="max_depth",
                    yaxis_title="Accuracy",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No numeric max_depth data available")
        
        st.markdown("---")
        
        # Model Registry Status
        st.subheader("🏆 Model Registry Status")
        
        try:
            versions = client.search_model_versions(f"name='churn-predictor'")
            
            if versions:
                registry_data = []
                for version in versions:
                    # Get run info for this version
                    run_id = version.run_id
                    run = client.get_run(run_id)
                    
                    # Determine status
                    aliases = version.aliases if hasattr(version, 'aliases') else []
                    if 'champion' in aliases or version.current_stage == 'Production':
                        status = "🏆 CHAMPION (Production)"
                        status_color = "#28a745"
                    elif 'challenger' in aliases or version.current_stage == 'Staging':
                        status = "🥈 CHALLENGER (Staging)"
                        status_color = "#ffc107"
                    else:
                        status = "📦 ARCHIVED"
                        status_color = "#6c757d"
                    
                    registry_data.append({
                        "Version": version.version,
                        "Status": status,
                        "Run ID": run_id[:8],
                        "Accuracy": run.data.metrics.get('test_accuracy', 'N/A'),
                        "Created": datetime.fromtimestamp(int(version.creation_timestamp) / 1000).strftime("%Y-%m-%d %H:%M"),
                    })
                
                reg_df = pd.DataFrame(registry_data)
                
                # Display with custom styling
                for _, row in reg_df.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 2])
                    
                    with col1:
                        st.markdown(f"**Version {row['Version']}**")
                    with col2:
                        st.markdown(f"{row['Status']}")
                    with col3:
                        st.markdown(f"`{row['Run ID']}`")
                    with col4:
                        if isinstance(row['Accuracy'], float):
                            st.markdown(f"**{row['Accuracy']:.1%}**")
                        else:
                            st.markdown(f"**{row['Accuracy']}**")
                    with col5:
                        st.markdown(f"*{row['Created']}*")
                
            else:
                st.info("📭 No models in registry yet. Register a model by running the training script with model registration enabled.")
                
        except Exception as e:
            st.warning(f"⚠️ Could not load model registry: {e}")
            st.info("Make sure you have registered at least one model version.")
        
        st.markdown("---")
        
        # Best Model Summary
        st.subheader("🌟 Champion Model Summary")
        
        best_run = df.loc[df['Accuracy'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Performance Metrics")
            st.metric("Accuracy", f"{best_run['Accuracy']:.3f}")
            st.metric("Precision", f"{best_run['Precision']:.3f}")
            st.metric("Recall", f"{best_run['Recall']:.3f}")
            st.metric("F1-Score", f"{best_run['F1-Score']:.3f}")
            st.metric("ROC-AUC", f"{best_run['ROC-AUC']:.3f}")
        
        with col2:
            st.markdown("### ⚙️ Hyperparameters")
            st.info(f"**n_estimators:** {best_run['n_estimators']}")
            st.info(f"**max_depth:** {best_run['max_depth']}")
            st.info(f"**min_samples_split:** {best_run['min_samples_split']}")
        
        with col3:
            st.markdown("### 📅 Details")
            st.success(f"**Run ID:** `{best_run['Run ID']}`")
            st.success(f"**Trained:** {best_run['Date']}")
            st.success(f"**Status:** {best_run['Status']}")
        
    else:
        st.error("❌ Experiment not found.")

except Exception as e:
    st.error(f"❌ Error loading MLflow data: {e}")
    st.info("""
    **Troubleshooting:**
    - Make sure you've run `train.py` at least once
    - Check that `mlflow.db` exists in the project root
    - Verify MLflow tracking URI is configured correctly
    """)
