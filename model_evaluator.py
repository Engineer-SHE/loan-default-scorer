"""
Model Evaluator
Provides comprehensive evaluation metrics for loan default prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)


class ModelEvaluator:
    """Evaluates loan default prediction models."""

    def __init__(self, scorer):
        """
        Initialize evaluator with a trained scorer.

        Parameters:
        -----------
        scorer : LoanDefaultScorer
            Trained loan scorer instance
        """
        self.scorer = scorer

    def evaluate(self, X_test, y_test, threshold=0.5, verbose=True):
        """
        Comprehensive model evaluation.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        threshold : float
            Classification threshold
        verbose : bool
            Whether to print results

        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Get predictions
        if self.scorer.model_type == 'ensemble':
            # Average predictions from all models
            probas = []
            for model in self.scorer.models.values():
                probas.append(model.predict_proba(X_test)[:, 1])
            y_proba = np.mean(probas, axis=0)
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_proba = self.scorer.model.predict_proba(X_test)[:, 1]
            y_pred = self.scorer.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        if verbose:
            self._print_metrics(metrics)

        return metrics

    def _print_metrics(self, metrics):
        """Print evaluation metrics in a formatted way."""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1 Score:          {metrics['f1_score']:.4f}")
        print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
        print(f"Avg Precision:     {metrics['avg_precision']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("="*50 + "\n")

    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """
        Plot confusion matrix.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        save_path : str, optional
            Path to save the plot
        """
        # Get predictions
        if self.scorer.model_type == 'ensemble':
            probas = []
            for model in self.scorer.models.values():
                probas.append(model.predict_proba(X_test)[:, 1])
            y_proba = np.mean(probas, axis=0)
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = self.scorer.model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ['No Default (0)', 'Default (1)'])
        plt.yticks([0.5, 1.5], ['No Default (0)', 'Default (1)'])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """
        Plot ROC curve.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        save_path : str, optional
            Path to save the plot
        """
        # Get probabilities
        if self.scorer.model_type == 'ensemble':
            probas = []
            for model in self.scorer.models.values():
                probas.append(model.predict_proba(X_test)[:, 1])
            y_proba = np.mean(probas, axis=0)
        else:
            y_proba = self.scorer.model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_precision_recall_curve(self, X_test, y_test, save_path=None):
        """
        Plot precision-recall curve.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        save_path : str, optional
            Path to save the plot
        """
        # Get probabilities
        if self.scorer.model_type == 'ensemble':
            probas = []
            for model in self.scorer.models.values():
                probas.append(model.predict_proba(X_test)[:, 1])
            y_proba = np.mean(probas, axis=0)
        else:
            y_proba = self.scorer.model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_feature_importance(self, top_n=15, save_path=None):
        """
        Plot feature importance.

        Parameters:
        -----------
        top_n : int
            Number of top features to show
        save_path : str, optional
            Path to save the plot
        """
        if self.scorer.feature_importance is None:
            print("Feature importance not available for this model type.")
            return

        importance = self.scorer.feature_importance
        indices = np.argsort(importance)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importance[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xticks(range(top_n), indices)
        plt.grid(alpha=0.3, axis='y')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_probability_distribution(self, X_test, y_test, save_path=None):
        """
        Plot distribution of predicted probabilities by true class.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        save_path : str, optional
            Path to save the plot
        """
        # Get probabilities
        if self.scorer.model_type == 'ensemble':
            probas = []
            for model in self.scorer.models.values():
                probas.append(model.predict_proba(X_test)[:, 1])
            y_proba = np.mean(probas, axis=0)
        else:
            y_proba = self.scorer.model.predict_proba(X_test)[:, 1]

        plt.figure(figsize=(10, 6))

        # Plot distributions for both classes
        plt.hist(y_proba[y_test == 0], bins=50, alpha=0.5, label='No Default (0)', color='green')
        plt.hist(y_proba[y_test == 1], bins=50, alpha=0.5, label='Default (1)', color='red')

        plt.xlabel('Predicted Default Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities by True Class')
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Probability distribution plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_classification_report(self, X_test, y_test):
        """
        Generate detailed classification report.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels

        Returns:
        --------
        str
            Classification report
        """
        # Get predictions
        if self.scorer.model_type == 'ensemble':
            probas = []
            for model in self.scorer.models.values():
                probas.append(model.predict_proba(X_test)[:, 1])
            y_proba = np.mean(probas, axis=0)
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = self.scorer.model.predict(X_test)

        report = classification_report(
            y_test, y_pred,
            target_names=['No Default', 'Default'],
            digits=4
        )

        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(report)
        print("="*50 + "\n")

        return report

    def evaluate_threshold_performance(self, X_test, y_test, thresholds=None):
        """
        Evaluate model performance across different classification thresholds.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        thresholds : list, optional
            List of thresholds to evaluate

        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        # Get probabilities
        if self.scorer.model_type == 'ensemble':
            probas = []
            for model in self.scorer.models.values():
                probas.append(model.predict_proba(X_test)[:, 1])
            y_proba = np.mean(probas, axis=0)
        else:
            y_proba = self.scorer.model.predict_proba(X_test)[:, 1]

        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            results.append({
                'threshold': threshold,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            })

        df_results = pd.DataFrame(results)
        print("\n" + "="*70)
        print("THRESHOLD PERFORMANCE ANALYSIS")
        print("="*70)
        print(df_results.to_string(index=False))
        print("="*70 + "\n")

        return df_results
