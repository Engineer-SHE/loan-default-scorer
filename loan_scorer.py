"""
Loan Default Risk Scorer
Main class for training and predicting loan default risk using machine learning.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from data_preprocessor import LoanDataPreprocessor


class LoanDefaultScorer:
    """
    Machine learning-powered loan default risk scorer.

    Supports multiple models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Ensemble (voting)
    """

    def __init__(self, model_type='xgboost', use_smote=True, random_state=42):
        """
        Initialize the loan scorer.

        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic', 'random_forest', 'xgboost', 'ensemble')
        use_smote : bool
            Whether to use SMOTE for handling class imbalance
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.random_state = random_state
        self.preprocessor = LoanDataPreprocessor()
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.training_score = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                scale_pos_weight=1,
                n_jobs=-1
            )
        elif self.model_type == 'ensemble':
            # Will create ensemble during training
            self.models = {
                'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'random_forest': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'xgboost': XGBClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, df, target_column='default', test_size=0.2, verbose=True):
        """
        Train the model on loan data.

        Parameters:
        -----------
        df : pd.DataFrame
            Training data with features and target
        target_column : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing
        verbose : bool
            Whether to print training progress

        Returns:
        --------
        dict
            Training results including scores
        """
        if verbose:
            print(f"Training {self.model_type} model...")
            print(f"Dataset size: {len(df)} samples")
            print(f"Default rate: {df[target_column].mean():.2%}")

        # Preprocess data
        X, y = self.preprocessor.fit_transform(df, target_column)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        if verbose:
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")

        # Apply SMOTE if enabled
        if self.use_smote and self.model_type != 'ensemble':
            if verbose:
                print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            if verbose:
                print(f"After SMOTE: {len(X_train)} samples")

        # Train model(s)
        if self.model_type == 'ensemble':
            if verbose:
                print("Training ensemble models...")
            for name, model in self.models.items():
                if verbose:
                    print(f"  Training {name}...")
                model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        self.is_trained = True

        # Calculate feature importance
        self._calculate_feature_importance()

        # Evaluate on test set
        from model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator(self)
        results = evaluator.evaluate(X_test, y_test, verbose=verbose)

        self.training_score = results

        return results

    def predict_proba(self, df, target_column='default'):
        """
        Predict default probability for new data.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to score (with or without target column)
        target_column : str
            Name of the target column (ignored if not present)

        Returns:
        --------
        np.ndarray
            Array of default probabilities (0.0 to 1.0)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")

        # Check if target column exists
        has_target = target_column in df.columns

        # Transform data
        X, _ = self.preprocessor.transform(df, target_column)

        # Get probabilities
        if self.model_type == 'ensemble':
            # Average predictions from all models
            probas = []
            for model in self.models.values():
                probas.append(model.predict_proba(X)[:, 1])
            y_proba = np.mean(probas, axis=0)
        else:
            y_proba = self.model.predict_proba(X)[:, 1]

        return y_proba

    def predict(self, df, threshold=0.5, target_column='default'):
        """
        Predict default classification (0 or 1).

        Parameters:
        -----------
        df : pd.DataFrame
            Data to score
        threshold : float
            Classification threshold (default: 0.5)
        target_column : str
            Name of the target column

        Returns:
        --------
        np.ndarray
            Array of predictions (0 or 1)
        """
        y_proba = self.predict_proba(df, target_column)
        return (y_proba >= threshold).astype(int)

    def score_loan(self, loan_data):
        """
        Score a single loan or batch of loans with risk categories.

        Parameters:
        -----------
        loan_data : dict or pd.DataFrame
            Single loan (dict) or multiple loans (DataFrame)

        Returns:
        --------
        dict or list of dict
            Risk score with probability and category
        """
        # Convert dict to DataFrame if needed
        if isinstance(loan_data, dict):
            df = pd.DataFrame([loan_data])
            single_loan = True
        else:
            df = loan_data
            single_loan = False

        # Get probabilities
        probabilities = self.predict_proba(df)

        # Create risk categories
        results = []
        for prob in probabilities:
            category = self._get_risk_category(prob)
            results.append({
                'default_probability': float(prob),
                'risk_score': float(prob * 100),
                'risk_category': category,
                'recommendation': self._get_recommendation(prob)
            })

        return results[0] if single_loan else results

    def _get_risk_category(self, probability):
        """Categorize risk based on probability."""
        if probability < 0.2:
            return 'LOW'
        elif probability < 0.4:
            return 'MEDIUM'
        elif probability < 0.6:
            return 'HIGH'
        else:
            return 'VERY_HIGH'

    def _get_recommendation(self, probability):
        """Provide lending recommendation based on probability."""
        if probability < 0.2:
            return 'APPROVE'
        elif probability < 0.4:
            return 'APPROVE_WITH_CONDITIONS'
        elif probability < 0.6:
            return 'MANUAL_REVIEW'
        else:
            return 'DECLINE'

    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        if self.model_type == 'logistic':
            # Use coefficients
            self.feature_importance = np.abs(self.model.coef_[0])
        elif self.model_type == 'random_forest':
            self.feature_importance = self.model.feature_importances_
        elif self.model_type == 'xgboost':
            self.feature_importance = self.model.feature_importances_
        elif self.model_type == 'ensemble':
            # Average importance from tree-based models
            importances = []
            if hasattr(self.models['random_forest'], 'feature_importances_'):
                importances.append(self.models['random_forest'].feature_importances_)
            if hasattr(self.models['xgboost'], 'feature_importances_'):
                importances.append(self.models['xgboost'].feature_importances_)
            if importances:
                self.feature_importance = np.mean(importances, axis=0)

    def get_feature_importance(self, top_n=10):
        """
        Get top N most important features.

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame
            DataFrame with features and importance scores
        """
        if self.feature_importance is None:
            return None

        # Note: This returns engineered feature importance
        # Feature names from preprocessor may not match exactly
        importance_df = pd.DataFrame({
            'feature_index': range(len(self.feature_importance)),
            'importance': self.feature_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath='loan_scorer_model.pkl'):
        """
        Save trained model to file.

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")

        model_data = {
            'model_type': self.model_type,
            'model': self.model if self.model_type != 'ensemble' else self.models,
            'preprocessor': self.preprocessor,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state,
            'use_smote': self.use_smote
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='loan_scorer_model.pkl'):
        """
        Load trained model from file.

        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model_type = model_data['model_type']
        if self.model_type == 'ensemble':
            self.models = model_data['model']
        else:
            self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state']
        self.use_smote = model_data['use_smote']
        self.is_trained = True

        print(f"Model loaded from {filepath}")

    def get_model_info(self):
        """Get information about the model."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'use_smote': self.use_smote,
            'random_state': self.random_state,
            'preprocessor_info': self.preprocessor.get_preprocessor_info()
        }
