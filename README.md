# 🔮 Telco Customer Churn Predictor (MLOps + Docker)

A production-ready MLOps project that predicts customer churn using an ensemble of **XGBoost**, **LightGBM**, and **CatBoost**.

## 🚀 Key Features
*   **High Accuracy**: **80.7% Accuracy** / **84.8% ROC-AUC** using an optimized voting ensemble.
*   **MLOps Pipeline**: Automated training, feature engineering, and model registration using **MLflow**.
*   **Dockerized**: Fully containerized application for "Write Once, Run Anywhere" deployment.
*   **Explainable AI**: Integrated **SHAP** values to explain *why* a customer is at risk.
*   **Interactive UI**: Built with **Streamlit** for real-time predictions and "What-If" analysis.

## 🛠️ Tech Stack
*   **Model**: XGBoost, LightGBM, CatBoost, Scikit-Learn
*   **Tracking**: MLflow
*   **App**: Streamlit, Plotly
*   **Container**: Docker
*   **Language**: Python 3.9

## 🐳 Quick Start (Docker)
The easiest way to run the app is using Docker.

1.  **Build the Image**:
    ```bash
    docker build -t churn-predictor .
    ```
2.  **Run the Container**:
    ```bash
    docker run -p 8501:8501 churn-predictor
    ```
3.  **Access the App**:
    Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📦 Manual Installation
If you prefer running locally without Docker:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Model** (Optional - Pre-trained model included):
    ```bash
    python train.py
    ```
3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
*   `app.py`: Main Streamlit application.
*   `train.py`: Training pipeline (Feature Eng -> Tuning -> Registration).
*   `utils/`: Helper modules for feature engineering.
*   `Dockerfile`: Container configuration.
*   `churn_pipeline.pkl`: Serialized model pipeline.
*   `mlruns/`: MLflow tracking data.

## 📈 Model Performance
*   **Accuracy**: 80.7%
*   **ROC-AUC**: 84.8%
*   **Optimal Threshold**: 0.50

See `improvements.md` for a detailed log of all enhancements.
