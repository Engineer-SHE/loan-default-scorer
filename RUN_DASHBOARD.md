# Running the Loan Default Risk Scorer Dashboard

## Quick Start

### 1. Install Dependencies

```bash
cd loan_default_scorer
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Dashboard Features

### 🏠 Home Page
- Overview of the loan scoring system
- Quick statistics about the model
- Key performance metrics

### 🎯 Score Individual Loan
- Interactive form to input loan application details
- Instant risk assessment with:
  - Default probability percentage
  - Risk score (0-100)
  - Risk category (LOW/MEDIUM/HIGH/VERY_HIGH)
  - Lending recommendation
- Visual risk gauge

### 📊 Model Performance
- Comprehensive evaluation metrics
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion matrix visualization
- ROC curve and Precision-Recall curve
- Feature importance analysis
- Threshold performance comparison

### 📈 Batch Analysis
- Generate and score multiple loan applications
- Portfolio-level risk distribution
- Risk category and recommendation breakdowns
- Interactive scatter plots showing:
  - Credit score vs default probability
  - Debt-to-income ratio vs default probability
- Sample loan data table

## Navigation

Use the sidebar to switch between different pages of the dashboard.

## Tips

- The model trains automatically on first load (takes about 30-60 seconds)
- All visualizations are interactive (hover, zoom, pan)
- Batch analysis allows you to generate different portfolio scenarios
- Try different loan parameters to see how risk scores change

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Ensure you're in the correct directory: `cd loan_default_scorer`
3. Check Python version (3.8+ recommended)
4. Clear Streamlit cache: Run the app and press 'C' in the terminal, then 'Clear Cache'

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the Streamlit server.
