"""
Sample Loan Data Generator
Generates synthetic loan data for testing the default risk scorer.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class LoanDataGenerator:
    """Generates realistic synthetic loan application data."""

    def __init__(self, seed=42):
        """Initialize the generator with a random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed

    def generate_data(self, n_samples=10000, default_rate=0.20):
        """
        Generate synthetic loan data.

        Parameters:
        -----------
        n_samples : int
            Number of loan applications to generate
        default_rate : float
            Target proportion of defaulted loans (0.0 to 1.0)

        Returns:
        --------
        pd.DataFrame
            DataFrame with loan features and default status
        """
        np.random.seed(self.seed)

        # Generate base features
        data = {}

        # Credit score (300-850, higher is better)
        data['credit_score'] = np.random.beta(5, 2, n_samples) * 550 + 300

        # Annual income ($20k - $200k)
        data['annual_income'] = np.random.lognormal(10.8, 0.6, n_samples)
        data['annual_income'] = np.clip(data['annual_income'], 20000, 200000)

        # Loan amount ($1k - $50k)
        data['loan_amount'] = np.random.lognormal(9.5, 0.7, n_samples)
        data['loan_amount'] = np.clip(data['loan_amount'], 1000, 50000)

        # Employment length (0-40 years)
        data['employment_length'] = np.random.exponential(5, n_samples)
        data['employment_length'] = np.clip(data['employment_length'], 0, 40)

        # Debt-to-income ratio (0-60%)
        data['debt_to_income'] = np.random.beta(2, 5, n_samples) * 60

        # Number of open credit lines (0-30)
        data['num_credit_lines'] = np.random.poisson(8, n_samples)
        data['num_credit_lines'] = np.clip(data['num_credit_lines'], 0, 30)

        # Number of derogatory marks (0-10)
        data['derogatory_marks'] = np.random.poisson(0.5, n_samples)
        data['derogatory_marks'] = np.clip(data['derogatory_marks'], 0, 10)

        # Total credit limit ($1k - $100k)
        data['total_credit_limit'] = data['annual_income'] * np.random.uniform(0.5, 2.5, n_samples)

        # Credit utilization (0-100%)
        data['credit_utilization'] = np.random.beta(2, 3, n_samples) * 100

        # Number of recent inquiries (0-10)
        data['recent_inquiries'] = np.random.poisson(1.5, n_samples)
        data['recent_inquiries'] = np.clip(data['recent_inquiries'], 0, 10)

        # Loan term (12, 36, or 60 months)
        data['loan_term'] = np.random.choice([12, 36, 60], n_samples, p=[0.2, 0.5, 0.3])

        # Interest rate (5-30%)
        data['interest_rate'] = np.random.beta(2, 5, n_samples) * 25 + 5

        # Home ownership (Rent, Own, Mortgage)
        data['home_ownership'] = np.random.choice(
            ['RENT', 'OWN', 'MORTGAGE'],
            n_samples,
            p=[0.35, 0.25, 0.40]
        )

        # Loan purpose
        data['loan_purpose'] = np.random.choice(
            ['debt_consolidation', 'credit_card', 'home_improvement',
             'major_purchase', 'medical', 'car', 'business', 'other'],
            n_samples,
            p=[0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]
        )

        # Number of historical defaults (0-5)
        data['historical_defaults'] = np.random.poisson(0.3, n_samples)
        data['historical_defaults'] = np.clip(data['historical_defaults'], 0, 5)

        # Months since last delinquency (0-120, or None)
        data['months_since_delinquency'] = np.random.exponential(24, n_samples)
        data['months_since_delinquency'] = np.where(
            np.random.rand(n_samples) > 0.3,
            data['months_since_delinquency'],
            np.nan
        )

        # Create DataFrame
        df = pd.DataFrame(data)

        # Generate default status based on risk factors
        default_probability = self._calculate_default_probability(df)

        # Adjust to match target default rate
        threshold = np.percentile(default_probability, (1 - default_rate) * 100)
        df['default'] = (default_probability > threshold).astype(int)

        # Add some noise
        flip_mask = np.random.rand(n_samples) < 0.05
        df.loc[flip_mask, 'default'] = 1 - df.loc[flip_mask, 'default']

        return df

    def _calculate_default_probability(self, df):
        """Calculate default probability based on risk factors."""
        prob = np.zeros(len(df))

        # Credit score (higher score = lower risk)
        prob += (850 - df['credit_score']) / 550 * 0.3

        # Debt-to-income ratio (higher = more risk)
        prob += df['debt_to_income'] / 60 * 0.2

        # Derogatory marks (more marks = more risk)
        prob += df['derogatory_marks'] / 10 * 0.15

        # Credit utilization (higher = more risk)
        prob += df['credit_utilization'] / 100 * 0.1

        # Historical defaults (more = more risk)
        prob += df['historical_defaults'] / 5 * 0.15

        # Recent inquiries (more = more risk)
        prob += df['recent_inquiries'] / 10 * 0.05

        # Employment length (shorter = more risk)
        prob += (40 - df['employment_length']) / 40 * 0.05

        # Normalize to [0, 1]
        prob = (prob - prob.min()) / (prob.max() - prob.min())

        return prob

    def save_data(self, df, filepath='loan_data.csv'):
        """Save generated data to CSV file."""
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        print(f"Total samples: {len(df)}")
        print(f"Default rate: {df['default'].mean():.2%}")


if __name__ == "__main__":
    # Generate sample data
    generator = LoanDataGenerator(seed=42)
    df = generator.generate_data(n_samples=10000, default_rate=0.20)

    # Display statistics
    print("\n=== Loan Data Summary ===")
    print(f"Total loans: {len(df)}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"\nFeature statistics:")
    print(df.describe())

    # Save data
    generator.save_data(df, 'loan_data.csv')
