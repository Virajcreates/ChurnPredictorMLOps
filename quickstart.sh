#!/bin/bash
# Quick Start Script for Churn Predictor MLOps Pipeline
# This script automates the setup and execution of the complete pipeline

set -e  # Exit on error

echo "🚀 Churn Predictor MLOps Pipeline - Quick Start"
echo "==============================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📋 NEXT STEPS:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "STEP 1: Train the model"
echo "   $ python train.py"
echo ""
echo "STEP 2: Start MLflow UI (in a new terminal)"
echo "   $ mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo "   Then open http://localhost:5000"
echo ""
echo "STEP 3: Promote model to Production"
echo "   • Go to Models tab → churn-predictor"
echo "   • Click Version 1 → Transition to Production"
echo ""
echo "STEP 4: Start the API server"
echo "   $ uvicorn serve:app --reload"
echo ""
echo "STEP 5: Test the API"
echo "   Open http://localhost:8000/docs"
echo ""
echo "═══════════════════════════════════════════════════════════"
