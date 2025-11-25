#!/usr/bin/env python3
"""
Hybrid AI Churn-Bot: Mission Control Dashboard
Combines MLOps (MLflow model) with Explainable AI for customer retention
"""
import json
import joblib
from pathlib import Path

import streamlit as st

# --- Configuration ---
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
from streamlit_extras.metric_cards import style_metric_cards
from rich.console import Console
import sys

# Import custom Feature Engineer
sys.path.append(str(Path(__file__).parent))
from utils.feature_engineering import FeatureEngineer

# Import prediction logging utility
try:
    from utils.prediction_logger import save_prediction_log
except (ImportError, ModuleNotFoundError):
    # Fallback if module not available
    def save_prediction_log(*args, **kwargs):
        """Dummy function when logger is not available"""
        pass


# Custom CSS for stunning styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Modern Card Styling */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25);
    }
    
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    .stAlert[data-baseweb="notification"] > div {
        border-radius: 12px;
    }
    
    /* Success/Stay - Emerald Green Theme */
    .stay-reason {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
        color: #065f46;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    
    /* Churn/Risk - Rose Red Theme */
    .churn-reason {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #ef4444;
        margin: 1rem 0;
        color: #991b1b;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    
    /* Explanation Box - Slate Gray */
    .explanation-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        color: #1e293b;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #cbd5e1;
    }
    
    /* Strategy Cards - Modern Professional */
    .strategy-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .strategy-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }
    
    .strategy-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .strategy-content {
        color: #475569;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    /* Factor Items - Clean Modern Style */
    .factor-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        color: #334155;
        border-left: 3px solid #3b82f6;
        transition: all 0.2s;
    }
    
    .factor-item:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateX(4px);
    }
    
    /* Modern Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 8px;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Metrics - Enhanced */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        color: #1e40af;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        color: #065f46;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }
    </style>
""", unsafe_allow_html=True)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_ROOT / "churn_pipeline.pkl"

# Simple model loading with joblib
@st.cache_resource
def load_model_pipeline():
    """Load the trained pipeline from churn_pipeline.pkl."""
    try:
        model = joblib.load(PIPELINE_PATH)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'churn_pipeline.pkl' not found!")
        st.info("""
        **To fix this:**
        1. Run `python train.py` locally to generate `churn_pipeline.pkl`
        2. Add, commit, and push `churn_pipeline.pkl` to your GitHub repository
        3. Redeploy your Streamlit app
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def get_shap_explainer(pipeline):
    """Create SHAP explainer for the model from the pipeline."""
    # Extract the classifier from the pipeline
    classifier = pipeline.named_steps['model']
    return shap.TreeExplainer(classifier)

def create_input_form():
    """Create the customer input form in the sidebar."""
    st.sidebar.header("📋 Customer Profile")
    
    with st.sidebar.form("customer_form"):
        st.subheader("Demographics")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with col2:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        col3, col4 = st.columns(2)
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
        with col4:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges, step=50.0)
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        else:
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
        
        st.subheader("Contract & Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    if submitted:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }
        return customer_data
    return None

def make_prediction(pipeline, customer_data):
    """Make churn prediction and get probability."""
    df = pd.DataFrame([customer_data])
    
    # The pipeline handles all preprocessing automatically
    proba = float(pipeline.predict_proba(df)[0][1])
    prediction = "Yes" if proba >= 0.5 else "No"
    
    return prediction, proba, df

def explain_prediction(pipeline, customer_data_df):
    """Generate SHAP explanation for the prediction using Ensemble."""
    # Extract steps
    # Pipeline: feature_engineering -> preprocessor -> smote (skipped) -> model
    
    # 1. Feature Engineering
    fe = pipeline.named_steps['feature_engineering']
    fe_data = fe.transform(customer_data_df)
    
    # 2. Preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    transformed_data = preprocessor.transform(fe_data)
    
    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()
    
    # 3. Model
    classifier = pipeline.named_steps['model']
    
    # Initialize aggregated SHAP values
    agg_shap_values = None
    
    # Iterate through each estimator in the VotingClassifier
    # classifier.estimators_ is a list of fitted sub-models
    estimators = classifier.estimators_
    
    valid_estimators_count = 0
    
    for model in estimators:
        try:
            # Create explainer for the sub-model
            if hasattr(model, 'get_booster'): # XGBoost
                explainer = shap.TreeExplainer(model.get_booster())
            elif hasattr(model, 'booster_'): # LightGBM
                explainer = shap.TreeExplainer(model.booster_)
            else: # CatBoost or others
                explainer = shap.TreeExplainer(model)
                
            # Calculate SHAP values
            shap_values = explainer.shap_values(transformed_data)
            
            # Handle different SHAP value shapes
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Binary classification positive class
            
            if len(shap_values.shape) == 3:
                shap_values = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:
                # If shape is (n_samples, n_features), take first sample
                if shap_values.shape[0] == 1:
                    shap_values = shap_values[0]
            
            # Add to aggregate
            if agg_shap_values is None:
                agg_shap_values = np.zeros_like(shap_values)
            
            agg_shap_values += shap_values
            valid_estimators_count += 1
            
        except Exception as e:
            print(f"⚠️ Could not explain model {type(model)}: {e}")
            continue
            
    if valid_estimators_count > 0:
        # Average the SHAP values
        avg_shap_values = agg_shap_values / valid_estimators_count
        # Return the last explainer just for the expected return signature (plotting uses values)
        return explainer, avg_shap_values, feature_names
    else:
        raise ValueError("Could not generate SHAP explanations for any ensemble model.")

def plot_shap_waterfall(explainer, shap_values, feature_names):
    """Create SHAP waterfall plot showing feature contributions."""
    import matplotlib.pyplot as plt
    
    # Create a bar plot instead of waterfall (simpler for transformed features)
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
    
    return fig

def generate_simple_explanation(customer_data, prediction, probability, top_risk_factors, top_protective_factors):
    """Generate a simple, human-readable explanation for why the customer will stay or leave."""
    
    if prediction == "Yes":
        # Customer will churn
        risk_summary = f"**Why This Customer is Likely to Leave:**\n\n"
        risk_summary += f"Based on our analysis, this customer has a **{probability:.0%} probability** of churning. "
        risk_summary += "Here are the main reasons:\n\n"
        
        for factor in top_risk_factors[:3]:
            if "Month-to-month" in str(factor):
                risk_summary += "🔴 **No long-term commitment**: They're on a month-to-month contract, making it easy to cancel anytime.\n\n"
            elif "tenure" in str(factor).lower() and customer_data.get('tenure', 0) < 12:
                risk_summary += f"🔴 **New customer**: Only {int(customer_data['tenure'])} months with us - haven't built loyalty yet.\n\n"
            elif "OnlineSecurity" in str(factor) and customer_data.get('OnlineSecurity') == 'No':
                risk_summary += "🔴 **Missing key services**: No online security protection adds to dissatisfaction.\n\n"
            elif "TechSupport" in str(factor) and customer_data.get('TechSupport') == 'No':
                risk_summary += "🔴 **No technical support**: When issues arise, they have no help, leading to frustration.\n\n"
            elif "Fiber optic" in str(factor) and customer_data.get('InternetService') == 'Fiber optic':
                risk_summary += "🔴 **Premium pricing concerns**: Fiber optic is expensive, and they may find cheaper alternatives.\n\n"
            elif "Electronic check" in str(factor):
                risk_summary += "🔴 **Manual payment method**: Electronic check is less convenient than automatic payments.\n\n"
        
        return risk_summary
    else:
        # Customer will stay
        stay_summary = f"**Why This Customer is Likely to Stay:**\n\n"
        stay_summary += f"Great news! This customer has only a **{probability:.0%} probability** of churning. "
        stay_summary += "Here's what's keeping them loyal:\n\n"
        
        for factor in top_protective_factors[:3]:
            if "tenure" in str(factor).lower() and customer_data.get('tenure', 0) > 24:
                stay_summary += f"✅ **Long-term loyalty**: {int(customer_data['tenure'])} months of service shows strong commitment.\n\n"
            elif "Two year" in str(factor) or "One year" in str(factor):
                stay_summary += f"✅ **Contract commitment**: Locked into a {customer_data.get('Contract', '')} contract - lower churn risk.\n\n"
            elif "OnlineSecurity" in str(factor) and customer_data.get('OnlineSecurity') == 'Yes':
                stay_summary += "✅ **Protected and secure**: Online security service adds value and peace of mind.\n\n"
            elif "TechSupport" in str(factor) and customer_data.get('TechSupport') == 'Yes':
                stay_summary += "✅ **Supported when needed**: Tech support subscription shows they value our assistance.\n\n"
            elif "automatic" in str(factor).lower():
                stay_summary += "✅ **Convenient payments**: Automatic payment method indicates satisfaction with service.\n\n"
            elif "Partner" in str(factor) and customer_data.get('Partner') == 'Yes':
                stay_summary += "✅ **Family plan benefits**: Partner on account suggests stable, family-oriented usage.\n\n"
        
        return stay_summary


def get_retention_strategy(customer_data, prediction, probability, top_factors):
    """Generate static retention strategy recommendations based on customer data."""
    
    risk_level = "HIGH RISK" if probability > 0.7 else "MODERATE RISK" if probability > 0.4 else "LOW RISK"
    tenure = customer_data['tenure']
    monthly_charges = customer_data['MonthlyCharges']
    
    # Determine customer segment
    if tenure < 12:
        segment = "New Customer (Under 1 Year)"
        segment_note = "Focus on early value demonstration and onboarding support"
    elif tenure >= 12 and tenure < 36:
        segment = "Growing Customer (1-3 Years)"
        segment_note = "Strengthen relationship with loyalty rewards"
    else:
        segment = "Loyal Long-Term Customer (3+ Years)"
        segment_note = "VIP treatment with exclusive appreciation offers"
    
    # Determine personalized offer based on risk and charges
    if probability > 0.7:
        if monthly_charges > 70:
            offer = "30% discount for 6 months + free premium support"
            urgency = "Immediate action required - Contact within 48 hours"
        else:
            offer = "3 months at 50% off + free service upgrade"
            urgency = "Immediate action required - Contact within 48 hours"
    elif probability > 0.4:
        offer = "20% discount for 3 months + complimentary tech support"
        urgency = "High priority - Contact within 1 week"
    else:
        offer = "15% loyalty discount + priority customer service"
        urgency = "Standard follow-up - Contact within 2 weeks"
    
    # Generate talking points based on top risk factors
    talking_points = []
    for factor in top_factors[:3]:
        factor_lower = factor.lower()
        if 'contract' in factor_lower:
            talking_points.append("Discuss flexible contract options and commitment benefits")
        elif 'charges' in factor_lower or 'price' in factor_lower:
            talking_points.append("Review current plan and identify cost-saving opportunities")
        elif 'support' in factor_lower or 'service' in factor_lower:
            talking_points.append("Highlight improved support channels and response times")
        elif 'internet' in factor_lower or 'fiber' in factor_lower:
            talking_points.append("Showcase latest internet speed upgrades and reliability improvements")
        elif 'payment' in factor_lower:
            talking_points.append("Offer convenient payment options and autopay discounts")
        else:
            talking_points.append(f"Address concerns about {factor}")
    
    # Determine contact method
    if probability > 0.6:
        contact_method = "Phone call"
        contact_note = "Personal touch required - Schedule immediate call"
    elif probability > 0.4:
        contact_method = "Email + Phone follow-up"
        contact_note = "Email first, then call if no response in 3 days"
    else:
        contact_method = "Email"
        contact_note = "Professional email communication sufficient"
    
    return {
        'segment': segment,
        'segment_note': segment_note,
        'risk_level': risk_level,
        'offer': offer,
        'urgency': urgency,
        'talking_points': talking_points,
        'contact_method': contact_method,
        'contact_note': contact_note,
        'tenure': tenure,
        'monthly_charges': monthly_charges
    }

def display_prediction_results(prediction, probability):
    """Display prediction results with visual indicators."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Prediction")
        if prediction == "Yes":
            st.error(f"**WILL CHURN** ⚠️")
            st.caption(f"Probability ≥ 50%")
        else:
            st.success(f"**WILL STAY** ✅")
            st.caption(f"Probability < 50%")
    
    with col2:
        st.markdown("### 📊 Confidence")
        st.metric("Churn Probability", f"{probability:.1%}")
        # Add interpretation helper
        if probability < 0.3:
            st.caption("Very low churn risk")
        elif probability < 0.5:
            st.caption("Low churn risk")
        elif probability < 0.7:
            st.caption("Moderate churn risk")
        else:
            st.caption("High churn risk")
    
    with col3:
        st.markdown("### 🚨 Risk Level")
        if probability > 0.7:
            st.error("**HIGH RISK**")
        elif probability > 0.4:
            st.warning("**MODERATE RISK**")
        else:
            st.success("**LOW RISK**")
    
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, width="stretch")

def main():
    """Main application logic."""
    # Header with animation
    st.markdown('<h1 class="main-header">🤖 Hybrid AI Churn-Bot: Mission Control</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Combining Predictive AI + Explainable AI</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load the model pipeline
    model = load_model_pipeline()
    
    # Display model info with style
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("🚀 **Active Model**: v1.0 (churn_pipeline.pkl) | **MLOps Stack**: MLflow + SHAP")
    
    # Sidebar input form
    customer_data = create_input_form()
    
    if customer_data is None:
        # Welcome screen with enhanced styling
        st.markdown("""
            ## 👋 Welcome to the Hybrid AI Churn-Bot!
            
            ### 🚀 How to Use:
            1. **Fill in** the customer profile form in the sidebar ←
            2. **Click** "🔮 Predict Churn" to get instant AI analysis.
            3. **Review** the prediction, the "why" from SHAP, and the AI-generated retention email.
            
            ### 💡 Try These Quick Test Scenarios:
            
            <p><strong>🔴 High Risk Customer (Will Probably Leave):</strong></p>
            <ul>
                <li><strong>Tenure</strong>: 1-2 months (very new)</li>
                <li><strong>Contract</strong>: Month-to-month (no commitment)</li>
                <li><strong>Monthly Charges</strong>: $80-100 (expensive)</li>
                <li><strong>Tech Support</strong>: No</li>
                <li><strong>Payment</strong>: Electronic check</li>
            </ul>
            
            <p><strong>🟢 Low Risk Customer (Will Stay):</strong></p>
            <ul>
                <li><strong>Tenure</strong>: 60+ months (loyal)</li>
                <li><strong>Contract</strong>: Two year (committed)</li>
                <li><strong>Tech Support</strong>: Yes</li>
                <li><strong>Payment</strong>: Bank transfer (automatic)</li>
            </ul>
        """, unsafe_allow_html=True)
        
        # Display metrics in cards
        st.markdown("### 📊 System Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Model Accuracy", "80.7%", help="Model accuracy on test set")
        with col2:
            st.metric("🔮 Predictions Made", "0", help="In this session")
        with col3:
            st.metric("⚡ Avg Response Time", "<2s", help="Prediction + Explanation + Strategy")
        with col4:
            st.metric("📊 Model Type", "Ensemble + SMOTE", help="XGB+LGBM+Cat + Feature Engineering")
        
    else:
        # Make prediction
        with st.spinner("🔮 Analyzing customer profile with AI..."):
            prediction, probability, customer_df = make_prediction(model, customer_data)
        
        # Log prediction for monitoring (production tracking)
        try:
            customer_id = f"CUST_{hash(str(customer_data))}"
            save_prediction_log(
                customer_id=customer_id,
                features=customer_data,
                prediction=prediction,
                probability=probability
            )
        except Exception as e:
            # Silently fail if logging doesn't work (don't break the app)
            pass
        
        # Display results with enhanced visuals
        st.markdown("## 🎯 Prediction Results")
        display_prediction_results(prediction, probability)
        
        st.markdown("---")
        
        # Explainability section
        st.markdown("## 🧠 AI Explainability: Understanding the Decision")
        
        with st.spinner("🔍 Generating SHAP explanation and natural language summary..."):
            explainer, shap_values, transformed_feature_names = explain_prediction(model, customer_df)
            
            # Get feature importance using transformed feature names
            if transformed_feature_names is not None:
                feature_names = transformed_feature_names
            else:
                feature_names = customer_df.columns.tolist()
            
            # shap_values is already 1D from explain_prediction, just ensure it's a proper numpy array
            shap_values_flat = np.asarray(shap_values).flatten()
            
            # Ensure lengths match
            if len(feature_names) != len(shap_values_flat):
                st.error(f"Feature count mismatch: {len(feature_names)} features but {len(shap_values_flat)} SHAP values")
                st.info(f"Debug: shap_values shape: {shap_values.shape}, feature_names length: {len(feature_names)}")
                return
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'Feature': list(feature_names),
                'Impact': list(shap_values_flat)
            }).sort_values('Impact', key=abs, ascending=False)
            
            # Map transformed features back to original features for better readability
            # Extract original feature names from transformed names (e.g., "Contract_Two year" -> "Contract")
            feature_importance['OriginalFeature'] = feature_importance['Feature'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )
            
            # Aggregate by original feature (sum absolute impacts)
            original_feature_importance = feature_importance.groupby('OriginalFeature').agg({
                'Impact': lambda x: x.sum()  # Sum the impacts
            }).reset_index()
            original_feature_importance.columns = ['Feature', 'Impact']
            original_feature_importance = original_feature_importance.sort_values('Impact', key=abs, ascending=False)
            
            # Get top factors from original features
            top_risk_factors = original_feature_importance[original_feature_importance['Impact'] > 0]['Feature'].head(5).tolist()
            top_protective_factors = original_feature_importance[original_feature_importance['Impact'] < 0]['Feature'].head(5).tolist()
            
            # Generate simple explanation
            simple_explanation = generate_simple_explanation(
                customer_data, prediction, probability, 
                top_risk_factors, top_protective_factors
            )
            
            # Display plain language explanation
            if prediction == "Yes":
                st.markdown(f'<div class="churn-reason">{simple_explanation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stay-reason">{simple_explanation}</div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Detailed Factor Analysis")
        
        # Display top factors in styled cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Factors Increasing Churn Risk")
            churn_factors = original_feature_importance[original_feature_importance['Impact'] > 0].head(5)
            max_impact = original_feature_importance['Impact'].abs().max()
            for idx, row in churn_factors.iterrows():
                st.markdown(f'<div class="factor-item">📍 <b>{row["Feature"]}</b>: +{row["Impact"]:.3f}</div>', unsafe_allow_html=True)
                st.progress(min(abs(row['Impact']) / max_impact, 1.0))
        
        with col2:
            st.markdown("#### 🟢 Factors Reducing Churn Risk")
            stay_factors = original_feature_importance[original_feature_importance['Impact'] < 0].head(5)
            for idx, row in stay_factors.iterrows():
                st.markdown(f'<div class="factor-item">📍 <b>{row["Feature"]}</b>: {row["Impact"]:.3f}</div>', unsafe_allow_html=True)
                st.progress(min(abs(row['Impact']) / feature_importance['Impact'].abs().max(), 1.0))
        
        # SHAP waterfall plot
        with st.expander("📊 Advanced: SHAP Waterfall Plot (Technical View)", expanded=False):
            st.info("This plot shows how each feature pushed the prediction away from the baseline.")
            fig = plot_shap_waterfall(explainer, shap_values, feature_names)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Retention Strategy Recommendations (for churn cases) or appreciation (for stay cases)
        if prediction == "Yes":
            st.markdown("## 📋 Recommended Retention Strategy")
            st.markdown("### 🎯 Actionable Plan Based on Risk Analysis")
            
            # Get retention strategy recommendations
            strategy = get_retention_strategy(
                customer_data, prediction, probability, top_risk_factors
            )
            
            # Display strategy in organized cards
            st.markdown("""
                <style>
                .strategy-card {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    border-left: 4px solid #3b82f6;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
                    margin-bottom: 1rem;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                
                .strategy-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
                }
                
                .strategy-header {
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: #1e293b;
                    margin-bottom: 0.75rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                .strategy-content {
                    color: #475569;
                    line-height: 1.7;
                    font-size: 0.95rem;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Customer Profile
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="strategy-card">
                    <div class="strategy-header">👤 Customer Profile</div>
                    <div class="strategy-content">
                        <b>Segment:</b> {strategy['segment']}<br>
                        <b>Tenure:</b> {strategy['tenure']} months<br>
                        <b>Monthly Charges:</b> ${strategy['monthly_charges']:.2f}<br>
                        <b>Risk Level:</b> {strategy['risk_level']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="strategy-card">
                    <div class="strategy-header">⏰ Urgency & Contact</div>
                    <div class="strategy-content">
                        <b>{strategy['urgency']}</b><br>
                        <b>Contact Method:</b> {strategy['contact_method']}<br>
                        <i>{strategy['contact_note']}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Personalized Offer
            st.markdown(f"""
            <div class="strategy-card" style="border-left-color: #2ca02c;">
                <div class="strategy-header" style="color: #2ca02c;">🎁 Personalized Offer</div>
                <div class="strategy-content">
                    <b>{strategy['offer']}</b><br>
                    <i>{strategy['segment_note']}</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Talking Points
            st.markdown(f"""
            <div class="strategy-card" style="border-left-color: #ff7f0e;">
                <div class="strategy-header" style="color: #ff7f0e;">💬 Key Talking Points</div>
                <div class="strategy-content">
                    {'<br>'.join([f'• {point}' for point in strategy['talking_points']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action Items
            st.markdown("""
            <div class="strategy-card" style="border-left-color: #d62728;">
                <div class="strategy-header" style="color: #d62728;">✅ Action Items</div>
                <div class="strategy-content">
                    1. Review customer account and recent interactions<br>
                    2. Prepare personalized offer details and approval<br>
                    3. Schedule contact using recommended method<br>
                    4. Follow up within 72 hours if no response<br>
                    5. Document outcome and update retention metrics
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Customer will stay - show appreciation message
            st.success("## ✅ Customer is Likely to Stay!")
            st.markdown("""
            <div class="stay-reason">
            
            ### 🎉 Great News!
            
            This customer shows strong loyalty indicators. Here's what to do:
            
            ✅ **Send a thank-you email** expressing appreciation for their continued business
            
            ✅ **Offer loyalty rewards** - Consider:
            - 10% discount on next bill
            - Free service upgrade for 3 months
            - Referral bonus ($50 credit for each friend they bring)
            
            ✅ **Proactive engagement** - Reach out 2 months before contract renewal with:
            - Exclusive early renewal offers
            - New service announcements
            - VIP customer appreciation events
            
            ✅ **Continue monitoring** - Set a reminder to check their usage in 3 months
            
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with updated branding
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 1.2rem; font-weight: 600;'>🤖 Hybrid AI Churn-Bot</p>
    <p style='font-size: 0.9rem;'>Powered by MLflow + SHAP</p>
    <p style='font-size: 0.85rem; color: #999;'>Combining Predictive AI + Explainable AI</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>⚡ Lightning-fast predictions • 🧠 Crystal-clear explanations • 📋 Actionable strategies</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
