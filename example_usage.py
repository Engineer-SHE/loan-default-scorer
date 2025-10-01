"""
Example Usage of Loan Default Risk Scorer
Demonstrates how to use the loan scoring system.
"""

import pandas as pd
import numpy as np
from sample_data_generator import LoanDataGenerator
from loan_scorer import LoanDefaultScorer
from model_evaluator import ModelEvaluator


def example_1_basic_training():
    """Example 1: Basic model training and evaluation."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Model Training")
    print("="*70 + "\n")

    # Generate sample data
    print("Step 1: Generating sample loan data...")
    generator = LoanDataGenerator(seed=42)
    df = generator.generate_data(n_samples=5000, default_rate=0.20)
    print(f"Generated {len(df)} loan applications")
    print(f"Default rate: {df['default'].mean():.2%}\n")

    # Create and train scorer
    print("Step 2: Training XGBoost model...")
    scorer = LoanDefaultScorer(model_type='xgboost', use_smote=True)
    results = scorer.train(df, verbose=True)

    # Save model
    print("\nStep 3: Saving model...")
    scorer.save_model('loan_scorer_xgboost.pkl')


def example_2_score_new_loans():
    """Example 2: Score new loan applications."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Scoring New Loan Applications")
    print("="*70 + "\n")

    # Load trained model
    print("Loading trained model...")
    scorer = LoanDefaultScorer()
    scorer.load_model('loan_scorer_xgboost.pkl')

    # Single loan application
    print("\n--- Scoring Single Loan ---")
    loan_application = {
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

    score = scorer.score_loan(loan_application)
    print(f"\nLoan Application Score:")
    print(f"  Default Probability: {score['default_probability']:.2%}")
    print(f"  Risk Score (0-100):  {score['risk_score']:.1f}")
    print(f"  Risk Category:       {score['risk_category']}")
    print(f"  Recommendation:      {score['recommendation']}")

    # Batch scoring
    print("\n--- Scoring Multiple Loans ---")
    new_loans = pd.DataFrame([
        {
            'credit_score': 720, 'annual_income': 85000, 'loan_amount': 10000,
            'employment_length': 8, 'debt_to_income': 20.0, 'num_credit_lines': 10,
            'derogatory_marks': 0, 'total_credit_limit': 75000, 'credit_utilization': 25.0,
            'recent_inquiries': 0, 'loan_term': 36, 'interest_rate': 10.5,
            'home_ownership': 'OWN', 'loan_purpose': 'home_improvement',
            'historical_defaults': 0, 'months_since_delinquency': np.nan
        },
        {
            'credit_score': 580, 'annual_income': 35000, 'loan_amount': 20000,
            'employment_length': 1, 'debt_to_income': 45.0, 'num_credit_lines': 5,
            'derogatory_marks': 3, 'total_credit_limit': 15000, 'credit_utilization': 85.0,
            'recent_inquiries': 5, 'loan_term': 60, 'interest_rate': 22.5,
            'home_ownership': 'RENT', 'loan_purpose': 'credit_card',
            'historical_defaults': 2, 'months_since_delinquency': 6.0
        }
    ])

    scores = scorer.score_loan(new_loans)
    for i, score in enumerate(scores, 1):
        print(f"\nLoan {i}:")
        print(f"  Risk Score: {score['risk_score']:.1f}")
        print(f"  Category:   {score['risk_category']}")
        print(f"  Decision:   {score['recommendation']}")


def example_3_compare_models():
    """Example 3: Compare different model types."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Different Models")
    print("="*70 + "\n")

    # Generate data
    generator = LoanDataGenerator(seed=42)
    df = generator.generate_data(n_samples=3000, default_rate=0.20)

    model_types = ['logistic', 'random_forest', 'xgboost', 'ensemble']
    results_summary = []

    for model_type in model_types:
        print(f"\n--- Training {model_type.upper()} ---")
        scorer = LoanDefaultScorer(model_type=model_type, use_smote=True)
        results = scorer.train(df, verbose=False)

        results_summary.append({
            'model': model_type,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'roc_auc': results['roc_auc']
        })

    # Display comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    comparison_df = pd.DataFrame(results_summary)
    print(comparison_df.to_string(index=False))
    print("="*70)


def example_4_advanced_evaluation():
    """Example 4: Advanced model evaluation with visualizations."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Advanced Model Evaluation")
    print("="*70 + "\n")

    # Generate data
    generator = LoanDataGenerator(seed=42)
    df = generator.generate_data(n_samples=5000, default_rate=0.20)

    # Train model
    print("Training model...")
    scorer = LoanDefaultScorer(model_type='xgboost', use_smote=True)
    scorer.train(df, verbose=False)

    # Prepare test data
    from sklearn.model_selection import train_test_split
    X, y = scorer.preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create evaluator
    evaluator = ModelEvaluator(scorer)

    # Generate classification report
    evaluator.generate_classification_report(X_test, y_test)

    # Evaluate different thresholds
    evaluator.evaluate_threshold_performance(X_test, y_test)

    # Show feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = scorer.get_feature_importance(top_n=10)
    if importance_df is not None:
        print(importance_df.to_string(index=False))

    # Note: Visualization plots can be generated but won't display in console
    print("\n[Visualization plots available via evaluator methods:]")
    print("  - evaluator.plot_confusion_matrix(X_test, y_test)")
    print("  - evaluator.plot_roc_curve(X_test, y_test)")
    print("  - evaluator.plot_precision_recall_curve(X_test, y_test)")
    print("  - evaluator.plot_feature_importance()")
    print("  - evaluator.plot_probability_distribution(X_test, y_test)")


def example_5_production_workflow():
    """Example 5: Complete production workflow."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Production Workflow")
    print("="*70 + "\n")

    # Step 1: Train and save model
    print("Step 1: Training production model...")
    generator = LoanDataGenerator(seed=42)
    training_data = generator.generate_data(n_samples=10000, default_rate=0.20)

    scorer = LoanDefaultScorer(model_type='ensemble', use_smote=True)
    scorer.train(training_data, verbose=False)
    scorer.save_model('production_model.pkl')
    print("Model trained and saved as 'production_model.pkl'")

    # Step 2: Load model in production
    print("\nStep 2: Loading model for production use...")
    production_scorer = LoanDefaultScorer()
    production_scorer.load_model('production_model.pkl')

    # Step 3: Score incoming applications
    print("\nStep 3: Scoring new applications...")
    new_application = {
        'credit_score': 650,
        'annual_income': 60000,
        'loan_amount': 18000,
        'employment_length': 3,
        'debt_to_income': 35.0,
        'num_credit_lines': 6,
        'derogatory_marks': 1,
        'total_credit_limit': 40000,
        'credit_utilization': 50.0,
        'recent_inquiries': 2,
        'loan_term': 48,
        'interest_rate': 15.5,
        'home_ownership': 'RENT',
        'loan_purpose': 'car',
        'historical_defaults': 0,
        'months_since_delinquency': np.nan
    }

    # Get score
    result = production_scorer.score_loan(new_application)

    print("\n--- Loan Decision ---")
    print(f"Applicant Credit Score: {new_application['credit_score']}")
    print(f"Loan Amount Requested:  ${new_application['loan_amount']:,.0f}")
    print(f"\nRisk Assessment:")
    print(f"  Default Probability: {result['default_probability']:.2%}")
    print(f"  Risk Score:          {result['risk_score']:.1f}/100")
    print(f"  Risk Category:       {result['risk_category']}")
    print(f"  Decision:            {result['recommendation']}")

    # Step 4: Business logic
    print("\n--- Business Decision Logic ---")
    if result['recommendation'] == 'APPROVE':
        print("✓ APPROVE: Low risk - proceed with standard terms")
    elif result['recommendation'] == 'APPROVE_WITH_CONDITIONS':
        print("⚠ APPROVE WITH CONDITIONS: Medium risk - consider higher interest rate or collateral")
    elif result['recommendation'] == 'MANUAL_REVIEW':
        print("⚠ MANUAL REVIEW REQUIRED: High risk - escalate to senior underwriter")
    else:
        print("✗ DECLINE: Very high risk - reject application")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LOAN DEFAULT RISK SCORER - EXAMPLE USAGE")
    print("="*70)

    # Run examples
    try:
        example_1_basic_training()
        example_2_score_new_loans()
        example_3_compare_models()
        example_4_advanced_evaluation()
        example_5_production_workflow()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
