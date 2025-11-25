import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Ensure numeric columns are numeric
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
        if 'tenure' in X.columns:
            X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce').fillna(0)
        if 'MonthlyCharges' in X.columns:
            X['MonthlyCharges'] = pd.to_numeric(X['MonthlyCharges'], errors='coerce').fillna(0)

        # 1. Tenure in Years
        if 'tenure' in X.columns:
            X['TenureYears'] = X['tenure'] / 12.0

        # 2. Average Monthly Charges (Total / Tenure)
        if 'TotalCharges' in X.columns and 'tenure' in X.columns:
            # Avoid division by zero
            X['AvgMonthlyCharges'] = np.where(
                X['tenure'] > 0, 
                X['TotalCharges'] / X['tenure'], 
                X['MonthlyCharges']
            )

        # 3. Has Support Services
        if 'TechSupport' in X.columns:
            X['HasSupport'] = (X['TechSupport'] == 'Yes').astype(int)
            
        # 4. Has Security Services
        if 'OnlineSecurity' in X.columns:
            X['HasSecurity'] = (X['OnlineSecurity'] == 'Yes').astype(int)

        # 5. Total Services Count
        services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        X['ServicesCount'] = 0
        for service in services:
            if service in X.columns:
                # Count 'Yes' or any valid service indicator (e.g. 'Fiber optic' for Internet)
                is_active = X[service].apply(lambda x: 1 if x not in ['No', 'No internet service', 'No phone service'] else 0)
                X['ServicesCount'] += is_active

        return X
