#!/usr/bin/env python3
"""
Production Monitoring Dashboard
Real-time monitoring of model predictions, drift detection, and performance tracking
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import logger utility
from utils.prediction_logger import load_prediction_logs, LOG_FILE

st.set_page_config(page_title="Production Monitor", page_icon="📡", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .alert-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .danger-box {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📡 Production Monitoring Dashboard")
st.markdown("**Real-time monitoring of model predictions in production**")
st.markdown("---")

# Load data
df_logs = load_prediction_logs()

if len(df_logs) > 0:
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    df_logs['date'] = df_logs['timestamp'].dt.date
    df_logs['hour'] = df_logs['timestamp'].dt.hour
    
    # Header metrics
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_predictions = len(df_logs)
        st.metric("Total Predictions", f"{total_predictions:,}", help="All-time prediction count")
    
    with col2:
        churn_rate = (df_logs['prediction'] == 'Yes').mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}", 
                 delta=f"{churn_rate - 0.27:.1%}" if churn_rate else None,
                 delta_color="inverse",
                 help="Percentage of customers predicted to churn")
    
    with col3:
        avg_probability = df_logs['probability'].mean()
        st.metric("Avg Probability", f"{avg_probability:.1%}",
                 help="Average churn probability across all predictions")
    
    with col4:
        today_count = len(df_logs[df_logs['date'] == datetime.now().date()])
        st.metric("Today's Predictions", today_count,
                 help="Predictions made today")
    
    with col5:
        high_risk = len(df_logs[df_logs['probability'] > 0.7])
        st.metric("High Risk Customers", high_risk,
                 delta="Urgent" if high_risk > 0 else None,
                 delta_color="off",
                 help="Customers with >70% churn probability")
    
    st.markdown("---")
    
    # Time-based analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Prediction Volume Over Time")
        
        # Daily prediction counts
        daily_counts = df_logs.groupby('date').size().reset_index(name='count')
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['count'],
            mode='lines+markers',
            name='Daily Predictions',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig1.update_layout(
            title="Daily Prediction Volume",
            xaxis_title="Date",
            yaxis_title="Number of Predictions",
            height=350,
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("🕐 Hourly Distribution")
        
        # Hourly distribution
        hourly_counts = df_logs.groupby('hour').size().reset_index(name='count')
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=hourly_counts['hour'],
            y=hourly_counts['count'],
            marker=dict(
                color=hourly_counts['count'],
                colorscale='Viridis',
                showscale=True
            ),
            text=hourly_counts['count'],
            textposition='outside'
        ))
        
        fig2.update_layout(
            title="Predictions by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Count",
            height=350,
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Probability distribution
    st.subheader("📊 Churn Probability Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df_logs['probability'],
            nbinsx=20,
            marker=dict(
                color='#667eea',
                line=dict(color='white', width=1)
            ),
            name='Probability Distribution'
        ))
        
        # Add threshold lines
        fig3.add_vline(x=0.5, line_dash="dash", line_color="red", 
                      annotation_text="Decision Threshold (50%)")
        fig3.add_vline(x=0.7, line_dash="dash", line_color="orange",
                      annotation_text="High Risk (70%)")
        
        fig3.update_layout(
            title="Distribution of Churn Probabilities",
            xaxis_title="Churn Probability",
            yaxis_title="Count",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk Breakdown")
        
        low_risk = len(df_logs[df_logs['probability'] < 0.4])
        medium_risk = len(df_logs[(df_logs['probability'] >= 0.4) & (df_logs['probability'] < 0.7)])
        high_risk = len(df_logs[df_logs['probability'] >= 0.7])
        
        risk_df = pd.DataFrame({
            'Risk Level': ['🟢 Low', '🟡 Medium', '🔴 High'],
            'Count': [low_risk, medium_risk, high_risk],
            'Percentage': [
                f"{low_risk/len(df_logs)*100:.1f}%",
                f"{medium_risk/len(df_logs)*100:.1f}%",
                f"{high_risk/len(df_logs)*100:.1f}%"
            ]
        })
        
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_df['Risk Level'],
            values=risk_df['Count'],
            hole=0.4,
            marker=dict(colors=['#28a745', '#ffc107', '#dc3545'])
        )])
        
        fig_pie.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Model Drift Detection
    st.subheader("🚨 Model Drift Detection")
    
    window_size = min(100, len(df_logs) // 2)
    
    if len(df_logs) >= window_size:
        # Calculate rolling statistics
        df_logs_sorted = df_logs.sort_values('timestamp').reset_index(drop=True)
        df_logs_sorted['rolling_mean'] = df_logs_sorted['probability'].rolling(window=window_size, min_periods=1).mean()
        df_logs_sorted['rolling_std'] = df_logs_sorted['probability'].rolling(window=window_size, min_periods=1).std()
        
        # Baseline (first window_size predictions)
        baseline_mean = df_logs_sorted['probability'].head(window_size).mean()
        baseline_std = df_logs_sorted['probability'].head(window_size).std()
        
        # Current window
        current_mean = df_logs_sorted['probability'].tail(window_size).mean()
        current_std = df_logs_sorted['probability'].tail(window_size).std()
        
        # Drift detection
        drift = abs(current_mean - baseline_mean)
        drift_threshold = 0.1  # 10%
        
        # Display drift metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Baseline Mean", f"{baseline_mean:.1%}",
                     help=f"Average probability in first {window_size} predictions")
        
        with col2:
            st.metric("Current Mean", f"{current_mean:.1%}",
                     delta=f"{current_mean - baseline_mean:+.1%}",
                     delta_color="inverse",
                     help=f"Average probability in last {window_size} predictions")
        
        with col3:
            drift_status = "🔴 DRIFT DETECTED" if drift > drift_threshold else "✅ No Drift"
            st.metric("Drift Status", drift_status,
                     delta=f"{drift:.1%}" if drift > drift_threshold else None,
                     delta_color="off")
        
        # Drift visualization
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=df_logs_sorted.index,
            y=df_logs_sorted['rolling_mean'],
            mode='lines',
            name='Rolling Mean',
            line=dict(color='#667eea', width=2)
        ))
        
        # Add confidence band
        fig4.add_trace(go.Scatter(
            x=df_logs_sorted.index,
            y=df_logs_sorted['rolling_mean'] + 2*df_logs_sorted['rolling_std'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig4.add_trace(go.Scatter(
            x=df_logs_sorted.index,
            y=df_logs_sorted['rolling_mean'] - 2*df_logs_sorted['rolling_std'],
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Add baseline
        fig4.add_hline(y=baseline_mean, line_dash="dash", line_color="red",
                      annotation_text=f"Baseline: {baseline_mean:.1%}")
        
        fig4.update_layout(
            title=f"Rolling Mean Probability (Window={window_size})",
            xaxis_title="Prediction Number",
            yaxis_title="Churn Probability",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Alert
        if drift > drift_threshold:
            st.markdown(f"""
            <div class="danger-box">
                <h4>⚠️ MODEL DRIFT ALERT</h4>
                <p>Current prediction distribution differs significantly from baseline!</p>
                <ul>
                    <li><b>Drift magnitude:</b> {drift:.1%}</li>
                    <li><b>Threshold:</b> {drift_threshold:.1%}</li>
                    <li><b>Recommendation:</b> Consider retraining the model with recent data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ Model is stable. Drift ({drift:.1%}) is below threshold ({drift_threshold:.1%})")
    
    else:
        st.info(f"ℹ️ Need at least {window_size} predictions for drift detection. Current: {len(df_logs)}")
    
    st.markdown("---")
    
    # Recent predictions
    st.subheader("🕐 Recent Predictions")
    
    n_recent = st.slider("Number of recent predictions to show", 10, 50, 20)
    
    recent = df_logs.tail(n_recent)[['timestamp', 'customer_id', 'prediction', 'probability']].copy()
    recent = recent.sort_values('timestamp', ascending=False)
    recent['probability'] = recent['probability'].apply(lambda x: f"{x:.1%}")
    recent['risk_level'] = recent.apply(
        lambda row: '🔴 High' if float(row['probability'].strip('%'))/100 > 0.7 
                    else ('🟡 Medium' if float(row['probability'].strip('%'))/100 > 0.4 else '🟢 Low'),
        axis=1
    )
    
    st.dataframe(recent, use_container_width=True, hide_index=True)

else:
    st.info("📭 No predictions logged yet!")
    st.markdown("### How to Enable Monitoring")
    
    st.markdown("""
    Add this code to your main `app.py` after making a prediction:
    
    ```python
    # Import at the top
    import sys
    sys.path.append('pages')
    from pages.production_monitor import save_prediction_log
    
    # After prediction
    save_prediction_log(
        customer_id=f"CUST_{hash(str(customer_data))}",
        features=customer_data,
        prediction=prediction,
        probability=probability,
        shap_values=shap_values
    )
    ```
    
    Start making predictions in the main app, and this dashboard will automatically populate!
    """)
    
    # Show example log structure
    st.markdown("### Expected Log Format")
    example_log = {
        "timestamp": "2025-11-16T10:30:00",
        "customer_id": "CUST_12345",
        "prediction": "Yes",
        "probability": 0.72,
        "features": {"tenure": 12, "Contract": "Month-to-month"},
        "shap_top_3": [0.15, -0.08, 0.05]
    }
    st.json(example_log)

# Export this function for use in main app
__all__ = ['save_prediction_log']
