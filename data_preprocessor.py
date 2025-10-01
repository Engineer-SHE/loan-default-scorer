"""
Data Preprocessor
Handles feature engineering, transformation, and preprocessing for loan data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class LoanDataPreprocessor:
    """Preprocesses loan data for machine learning models."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.numerical_features = []
        self.categorical_features = []
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_numerical = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False

    def fit(self, df, target_column='default'):
        """
        Fit the preprocessor on training data.

        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        target_column : str
            Name of the target column

        Returns:
        --------
        self
        """
        # Separate features and target
        X = df.drop(columns=[target_column])

        # Identify numerical and categorical features
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fit imputers
        if self.numerical_features:
            self.imputer_numerical.fit(X[self.numerical_features])

        if self.categorical_features:
            self.imputer_categorical.fit(X[self.categorical_features])

        # Fit label encoders for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            # Fill NaN temporarily for fitting
            temp_data = X[col].fillna('MISSING')
            le.fit(temp_data)
            self.label_encoders[col] = le

        # Transform data to fit scaler
        X_transformed = self._transform_features(X)

        # Fit scaler
        self.scaler.fit(X_transformed)

        # Store feature names
        self.feature_names = self._get_feature_names()

        self.is_fitted = True
        return self

    def transform(self, df, target_column='default'):
        """
        Transform data using fitted preprocessor.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform
        target_column : str
            Name of the target column (if present)

        Returns:
        --------
        X_scaled : np.ndarray
            Scaled feature matrix
        y : np.ndarray or None
            Target variable (if present in df)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")

        # Check if target exists
        has_target = target_column in df.columns

        # Separate features and target
        if has_target:
            X = df.drop(columns=[target_column])
            y = df[target_column].values
        else:
            X = df.copy()
            y = None

        # Transform features
        X_transformed = self._transform_features(X)

        # Scale features
        X_scaled = self.scaler.transform(X_transformed)

        return X_scaled, y

    def fit_transform(self, df, target_column='default'):
        """
        Fit and transform data in one step.

        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        target_column : str
            Name of the target column

        Returns:
        --------
        X_scaled : np.ndarray
            Scaled feature matrix
        y : np.ndarray
            Target variable
        """
        self.fit(df, target_column)
        return self.transform(df, target_column)

    def _transform_features(self, X):
        """Transform features (imputation, encoding, engineering)."""
        X_copy = X.copy()

        # Impute missing values for numerical features
        if self.numerical_features:
            X_copy[self.numerical_features] = self.imputer_numerical.transform(
                X_copy[self.numerical_features]
            )

        # Impute and encode categorical features
        if self.categorical_features:
            X_copy[self.categorical_features] = self.imputer_categorical.transform(
                X_copy[self.categorical_features]
            )

            for col in self.categorical_features:
                le = self.label_encoders[col]
                # Handle unknown categories
                X_copy[col] = X_copy[col].apply(
                    lambda x: x if x in le.classes_ else 'MISSING'
                )
                X_copy[col] = le.transform(X_copy[col])

        # Engineer additional features
        X_engineered = self._engineer_features(X_copy)

        return X_engineered.values

    def _engineer_features(self, X):
        """Create additional features from existing ones."""
        X_copy = X.copy()

        # Loan amount to income ratio
        if 'loan_amount' in X_copy.columns and 'annual_income' in X_copy.columns:
            X_copy['loan_to_income_ratio'] = X_copy['loan_amount'] / (X_copy['annual_income'] + 1)

        # Credit limit utilization ratio
        if 'total_credit_limit' in X_copy.columns and 'annual_income' in X_copy.columns:
            X_copy['credit_limit_to_income'] = X_copy['total_credit_limit'] / (X_copy['annual_income'] + 1)

        # Average credit per line
        if 'total_credit_limit' in X_copy.columns and 'num_credit_lines' in X_copy.columns:
            X_copy['avg_credit_per_line'] = X_copy['total_credit_limit'] / (X_copy['num_credit_lines'] + 1)

        # Monthly payment estimate (simple calculation)
        if 'loan_amount' in X_copy.columns and 'interest_rate' in X_copy.columns and 'loan_term' in X_copy.columns:
            monthly_rate = X_copy['interest_rate'] / 100 / 12
            n_payments = X_copy['loan_term']
            X_copy['monthly_payment'] = (
                X_copy['loan_amount'] * monthly_rate * (1 + monthly_rate) ** n_payments /
                ((1 + monthly_rate) ** n_payments - 1)
            )
            # Handle edge cases
            X_copy['monthly_payment'] = X_copy['monthly_payment'].fillna(X_copy['loan_amount'] / X_copy['loan_term'])

        # Payment to income ratio
        if 'monthly_payment' in X_copy.columns and 'annual_income' in X_copy.columns:
            X_copy['payment_to_income'] = (X_copy['monthly_payment'] * 12) / (X_copy['annual_income'] + 1)

        # Risk score based on derogatory marks and inquiries
        if 'derogatory_marks' in X_copy.columns and 'recent_inquiries' in X_copy.columns:
            X_copy['credit_risk_score'] = X_copy['derogatory_marks'] * 2 + X_copy['recent_inquiries']

        # Has recent delinquency flag
        if 'months_since_delinquency' in X_copy.columns:
            X_copy['has_recent_delinquency'] = (
                X_copy['months_since_delinquency'].notna() &
                (X_copy['months_since_delinquency'] < 24)
            ).astype(int)

        # Replace any remaining NaN/inf values
        X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
        X_copy = X_copy.fillna(0)

        return X_copy

    def _get_feature_names(self):
        """Get list of all feature names after transformation."""
        # Create a dummy dataframe with one row to get feature names
        return self.numerical_features + self.categorical_features

    def get_feature_names(self):
        """Return the list of feature names."""
        return self.feature_names

    def inverse_transform_target(self, y_pred):
        """
        Convert predictions back to original format.

        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted values

        Returns:
        --------
        np.ndarray
            Predictions in original format
        """
        return y_pred

    def get_preprocessor_info(self):
        """Get information about the preprocessor configuration."""
        return {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'total_features': len(self.numerical_features) + len(self.categorical_features),
            'is_fitted': self.is_fitted
        }
