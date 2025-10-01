# Loan Default Risk Scorer

A machine learning-powered system to predict the probability that a borrower will default on a loan.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and Ensemble
- **Comprehensive Preprocessing**: Handles missing values, categorical encoding, feature scaling, and feature engineering
- **Risk Scoring**: Outputs probability scores (0-1) and risk categories (LOW/MEDIUM/HIGH/VERY_HIGH)
- **Production-Ready**: Model persistence, input validation, and error handling
- **Evaluation Tools**: ROC-AUC, precision/recall, confusion matrix, feature importance
- **Sample Data Generator**: Creates realistic synthetic loan data for testing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

```python
from sample_data_generator import LoanDataGenerator
from loan_scorer import LoanDefaultScorer

# Generate training data
generator = LoanDataGenerator(seed=42)
df = generator.generate_data(n_samples=10000, default_rate=0.20)

# Train model
scorer = LoanDefaultScorer(model_type='xgboost', use_smote=True)
scorer.train(df)

# Save model
scorer.save_model('my_model.pkl')
```

### 2. Score New Loans

```python
import numpy as np

# Load trained model
scorer = LoanDefaultScorer()
scorer.load_model('my_model.pkl')

# Score a single loan application
loan = {
    'credit_score': 680,
    'annual_income': 75000,
    'loan_amount': 15000,
    'employment_length': 5,
    'debt_to_income': 25.5,
    'num_credit_lines': 8,
    'derogatory_marks': 0,
    'total_credit_limit': 50000,
    'credit_utilization': 30.0,
    'recent_inquiries': 1,
    'loan_term': 36,
    'interest_rate': 12.5,
    'home_ownership': 'MORTGAGE',
    'loan_purpose': 'debt_consolidation',
    'historical_defaults': 0,
    'months_since_delinquency': np.nan
}

result = scorer.score_loan(loan)
print(f"Default Probability: {result['default_probability']:.2%}")
print(f"Risk Category: {result['risk_category']}")
print(f"Recommendation: {result['recommendation']}")
```

## Model Types

- **logistic**: Logistic Regression (fast, interpretable)
- **random_forest**: Random Forest (balanced performance)
- **xgboost**: XGBoost (best accuracy)
- **ensemble**: Voting ensemble of all three (most robust)

## Risk Categories

- **LOW** (0-20%): APPROVE
- **MEDIUM** (20-40%): APPROVE_WITH_CONDITIONS
- **HIGH** (40-60%): MANUAL_REVIEW
- **VERY_HIGH** (60-100%): DECLINE

## Loan Features

The model expects the following features:
- `credit_score`: Credit score (300-850)
- `annual_income`: Annual income in dollars
- `loan_amount`: Requested loan amount
- `employment_length`: Years of employment
- `debt_to_income`: Debt-to-income ratio (%)
- `num_credit_lines`: Number of open credit lines
- `derogatory_marks`: Number of derogatory marks
- `total_credit_limit`: Total credit limit
- `credit_utilization`: Credit utilization (%)
- `recent_inquiries`: Number of recent credit inquiries
- `loan_term`: Loan term in months
- `interest_rate`: Interest rate (%)
- `home_ownership`: RENT, OWN, or MORTGAGE
- `loan_purpose`: Purpose of the loan
- `historical_defaults`: Number of historical defaults
- `months_since_delinquency`: Months since last delinquency (or NaN)

## Examples

Run the comprehensive example script:

```bash
python example_usage.py
```

This demonstrates:
- Basic model training
- Scoring new loans
- Comparing different models
- Advanced evaluation metrics
- Production workflow

## File Structure

```
loan_default_scorer/
├── loan_scorer.py              # Main scoring class
├── data_preprocessor.py        # Feature engineering & preprocessing
├── model_evaluator.py          # Model evaluation metrics
├── sample_data_generator.py    # Synthetic data generator
├── example_usage.py            # Usage examples
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Advanced Usage

### Evaluate Model Performance

```python
from model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(scorer)
evaluator.generate_classification_report(X_test, y_test)
evaluator.plot_roc_curve(X_test, y_test, save_path='roc_curve.png')
evaluator.plot_confusion_matrix(X_test, y_test)
```

### Feature Importance

```python
importance = scorer.get_feature_importance(top_n=10)
print(importance)
```

### Compare Models

```python
models = ['logistic', 'random_forest', 'xgboost', 'ensemble']
for model_type in models:
    scorer = LoanDefaultScorer(model_type=model_type)
    results = scorer.train(df)
    print(f"{model_type}: ROC-AUC = {results['roc_auc']:.4f}")
```

## License

MIT License
