"""
Loan Default Risk Scorer - Streamlit Dashboard
Interactive dashboard for visualizing and using the loan scoring system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Import loan scorer modules
from sample_data_generator import LoanDataGenerator
from loan_scorer import LoanDefaultScorer
from model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Scorer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-very-high {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'scorer' not in st.session_state:
    st.session_state.scorer = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None


def train_model_if_needed():
    """Train model if not already trained."""
    if not st.session_state.model_trained:
        with st.spinner("Training model... This may take a minute."):
            # Generate data
            generator = LoanDataGenerator(seed=42)
            df = generator.generate_data(n_samples=5000, default_rate=0.20)

            # Split for later evaluation
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['default'])

            # Train model
            scorer = LoanDefaultScorer(model_type='xgboost', use_smote=True)
            scorer.train(train_df, verbose=False)

            # Save to session state
            st.session_state.scorer = scorer
            st.session_state.training_data = train_df
            st.session_state.test_data = test_df
            st.session_state.model_trained = True

        st.success("Model trained successfully!")


def home_page():
    """Display home page with overview."""
    st.markdown('<p class="main-header">💰 Loan Default Risk Scorer</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Loan Default Risk Scorer Dashboard

    This interactive dashboard showcases a machine learning-powered system that predicts
    the probability of loan default based on borrower characteristics.

    #### Features:
    - 🎯 **Score Individual Loans**: Get instant risk assessments for loan applications
    - 📊 **Model Performance**: Visualize model accuracy, ROC curves, and feature importance
    - 📈 **Batch Analysis**: Analyze multiple loans and view risk distribution
    - 🔍 **Data Exploration**: Explore synthetic loan data and patterns

    #### How It Works:
    1. The system uses **XGBoost** machine learning algorithm
    2. Analyzes 16+ features including credit score, income, debt ratios, and credit history
    3. Outputs risk probability (0-100%) and risk category
    4. Provides lending recommendations (APPROVE, REVIEW, or DECLINE)

    #### Navigation:
    Use the sidebar to navigate between different sections of the dashboard.
    """)

    # Quick stats
    train_model_if_needed()

    if st.session_state.model_trained:
        st.markdown("---")
        st.subheader("Quick Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Training Samples", f"{len(st.session_state.training_data):,}")

        with col2:
            default_rate = st.session_state.training_data['default'].mean()
            st.metric("Default Rate", f"{default_rate:.1%}")

        with col3:
            if st.session_state.scorer.training_score:
                roc_auc = st.session_state.scorer.training_score['roc_auc']
                st.metric("ROC-AUC Score", f"{roc_auc:.3f}")

        with col4:
            if st.session_state.scorer.training_score:
                accuracy = st.session_state.scorer.training_score['accuracy']
                st.metric("Accuracy", f"{accuracy:.1%}")


def score_loan_page():
    """Interactive loan scoring interface."""
    st.header("🎯 Score Individual Loan Application")

    train_model_if_needed()

    st.markdown("""
    Enter the details of a loan application below to get an instant risk assessment.
    """)

    # Create input form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Borrower Information")
            credit_score = st.slider("Credit Score", 300, 850, 680, 10)
            annual_income = st.number_input("Annual Income ($)", 20000, 200000, 75000, 5000)
            employment_length = st.slider("Employment Length (years)", 0, 40, 5, 1)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

            st.subheader("Credit History")
            num_credit_lines = st.slider("Number of Credit Lines", 0, 30, 8, 1)
            derogatory_marks = st.slider("Derogatory Marks", 0, 10, 0, 1)
            historical_defaults = st.slider("Historical Defaults", 0, 5, 0, 1)
            recent_inquiries = st.slider("Recent Inquiries", 0, 10, 1, 1)

        with col2:
            st.subheader("Loan Details")
            loan_amount = st.number_input("Loan Amount ($)", 1000, 50000, 15000, 1000)
            loan_term = st.selectbox("Loan Term (months)", [12, 36, 60])
            interest_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.5, 0.5)
            loan_purpose = st.selectbox("Loan Purpose", [
                "debt_consolidation", "credit_card", "home_improvement",
                "major_purchase", "medical", "car", "business", "other"
            ])

            st.subheader("Financial Metrics")
            debt_to_income = st.slider("Debt-to-Income Ratio (%)", 0.0, 60.0, 25.5, 0.5)
            total_credit_limit = st.number_input("Total Credit Limit ($)", 1000, 100000, 50000, 1000)
            credit_utilization = st.slider("Credit Utilization (%)", 0.0, 100.0, 30.0, 1.0)
            months_since_delinquency = st.number_input(
                "Months Since Last Delinquency (blank if none)",
                min_value=0, max_value=120, value=None, step=1
            )

        submit_button = st.form_submit_button("Calculate Risk Score", use_container_width=True)

    if submit_button:
        # Create loan data
        loan_data = {
            'credit_score': credit_score,
            'annual_income': annual_income,
            'loan_amount': loan_amount,
            'employment_length': employment_length,
            'debt_to_income': debt_to_income,
            'num_credit_lines': num_credit_lines,
            'derogatory_marks': derogatory_marks,
            'total_credit_limit': total_credit_limit,
            'credit_utilization': credit_utilization,
            'recent_inquiries': recent_inquiries,
            'loan_term': loan_term,
            'interest_rate': interest_rate,
            'home_ownership': home_ownership,
            'loan_purpose': loan_purpose,
            'historical_defaults': historical_defaults,
            'months_since_delinquency': np.nan if months_since_delinquency is None else float(months_since_delinquency)
        }

        # Score the loan
        result = st.session_state.scorer.score_loan(loan_data)

        # Display results
        st.markdown("---")
        st.subheader("Risk Assessment Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Default Probability", f"{result['default_probability']:.1%}")

        with col2:
            st.metric("Risk Score (0-100)", f"{result['risk_score']:.1f}")

        with col3:
            risk_class = f"risk-{result['risk_category'].lower().replace('_', '-')}"
            st.markdown(f"**Risk Category**")
            st.markdown(f'<p class="{risk_class}" style="font-size: 1.5rem;">{result["risk_category"]}</p>',
                       unsafe_allow_html=True)

        # Recommendation
        st.markdown("---")
        recommendation = result['recommendation']

        if recommendation == 'APPROVE':
            st.success(f"✅ **Recommendation: {recommendation}**")
            st.info("Low risk applicant. Proceed with standard lending terms.")
        elif recommendation == 'APPROVE_WITH_CONDITIONS':
            st.warning(f"⚠️ **Recommendation: {recommendation}**")
            st.info("Medium risk applicant. Consider higher interest rate or additional collateral.")
        elif recommendation == 'MANUAL_REVIEW':
            st.warning(f"⚠️ **Recommendation: {recommendation}**")
            st.info("High risk applicant. Escalate to senior underwriter for manual review.")
        else:
            st.error(f"❌ **Recommendation: {recommendation}**")
            st.info("Very high risk applicant. Consider declining the application.")

        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result['risk_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'size': 24}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#28a745'},
                    {'range': [20, 40], 'color': '#ffc107'},
                    {'range': [40, 60], 'color': '#fd7e14'},
                    {'range': [60, 100], 'color': '#dc3545'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def model_performance_page():
    """Display model performance metrics and visualizations."""
    st.header("📊 Model Performance Analysis")

    train_model_if_needed()

    scorer = st.session_state.scorer
    test_df = st.session_state.test_data

    # Get test data
    X_test, y_test = scorer.preprocessor.transform(test_df)

    # Create evaluator
    evaluator = ModelEvaluator(scorer)
    metrics = evaluator.evaluate(X_test, y_test, verbose=False)

    # Display metrics
    st.subheader("Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    with col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("---")

    # Confusion Matrix
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: No Default', 'Predicted: Default'],
            y=['Actual: No Default', 'Actual: Default'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Blues',
            showscale=True
        ))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Performance by Threshold")

        # Calculate metrics for different thresholds
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []

        y_proba = scorer.model.predict_proba(X_test)[:, 1]

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            threshold_metrics.append({
                'Threshold': threshold,
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, zero_division=0)
            })

        df_threshold = pd.DataFrame(threshold_metrics)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_threshold['Threshold'], y=df_threshold['Precision'],
                                name='Precision', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df_threshold['Threshold'], y=df_threshold['Recall'],
                                name='Recall', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df_threshold['Threshold'], y=df_threshold['F1 Score'],
                                name='F1 Score', mode='lines+markers'))

        fig.update_layout(
            xaxis_title='Classification Threshold',
            yaxis_title='Score',
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ROC and PR Curves
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")

        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {metrics["roc_auc"]:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Precision-Recall Curve")

        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR (AP = {metrics["avg_precision"]:.3f})',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))

        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.subheader("Feature Importance (Top 15)")

    if scorer.feature_importance is not None:
        importance = scorer.feature_importance
        indices = np.argsort(importance)[::-1][:15]

        fig = go.Figure(go.Bar(
            x=importance[indices],
            y=[f'Feature {i}' for i in indices],
            orientation='h',
            marker=dict(color=importance[indices], colorscale='Viridis')
        ))

        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


def batch_analysis_page():
    """Batch scoring and portfolio analysis."""
    st.header("📈 Batch Loan Analysis")

    train_model_if_needed()

    st.markdown("""
    Generate and analyze a portfolio of loan applications to understand risk distribution.
    """)

    # Generate sample loans
    col1, col2 = st.columns([1, 3])

    with col1:
        n_loans = st.slider("Number of Loans to Generate", 100, 2000, 500, 100)
        default_rate = st.slider("Target Default Rate (%)", 10, 40, 20, 5) / 100

    if st.button("Generate Loan Portfolio", use_container_width=True):
        with st.spinner("Generating and scoring loans..."):
            # Generate data
            generator = LoanDataGenerator(seed=np.random.randint(1000))
            df = generator.generate_data(n_samples=n_loans, default_rate=default_rate)

            # Score all loans
            scores = st.session_state.scorer.score_loan(df.drop(columns=['default']))
            df['predicted_probability'] = [s['default_probability'] for s in scores]
            df['predicted_category'] = [s['risk_category'] for s in scores]
            df['recommendation'] = [s['recommendation'] for s in scores]

            # Store in session state
            st.session_state.batch_df = df

        st.success(f"Generated and scored {n_loans} loans!")

    if 'batch_df' in st.session_state:
        df = st.session_state.batch_df

        st.markdown("---")
        st.subheader("Portfolio Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Loans", f"{len(df):,}")
        with col2:
            approve_rate = (df['recommendation'] == 'APPROVE').sum() / len(df)
            st.metric("Approve Rate", f"{approve_rate:.1%}")
        with col3:
            decline_rate = (df['recommendation'] == 'DECLINE').sum() / len(df)
            st.metric("Decline Rate", f"{decline_rate:.1%}")
        with col4:
            avg_risk = df['predicted_probability'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.1%}")

        # Risk distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Category Distribution")

            category_counts = df['predicted_category'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=.3,
                marker=dict(colors=['#28a745', '#ffc107', '#fd7e14', '#dc3545'])
            )])

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Recommendation Distribution")

            rec_counts = df['recommendation'].value_counts()

            fig = go.Figure(data=[go.Bar(
                x=rec_counts.index,
                y=rec_counts.values,
                marker_color=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
            )])

            fig.update_layout(
                xaxis_title='Recommendation',
                yaxis_title='Count',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Risk score distribution
        st.subheader("Predicted Default Probability Distribution")

        fig = go.Figure()

        # Separate by actual default status
        fig.add_trace(go.Histogram(
            x=df[df['default'] == 0]['predicted_probability'],
            name='No Default (Actual)',
            opacity=0.7,
            nbinsx=30,
            marker_color='green'
        ))

        fig.add_trace(go.Histogram(
            x=df[df['default'] == 1]['predicted_probability'],
            name='Default (Actual)',
            opacity=0.7,
            nbinsx=30,
            marker_color='red'
        ))

        fig.update_layout(
            xaxis_title='Predicted Default Probability',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature correlations
        st.subheader("Risk Factors Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Credit score vs risk
            fig = px.scatter(
                df.sample(min(500, len(df))),
                x='credit_score',
                y='predicted_probability',
                color='predicted_category',
                title='Credit Score vs Default Probability',
                color_discrete_map={
                    'LOW': '#28a745',
                    'MEDIUM': '#ffc107',
                    'HIGH': '#fd7e14',
                    'VERY_HIGH': '#dc3545'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Debt-to-income vs risk
            fig = px.scatter(
                df.sample(min(500, len(df))),
                x='debt_to_income',
                y='predicted_probability',
                color='predicted_category',
                title='Debt-to-Income Ratio vs Default Probability',
                color_discrete_map={
                    'LOW': '#28a745',
                    'MEDIUM': '#ffc107',
                    'HIGH': '#fd7e14',
                    'VERY_HIGH': '#dc3545'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("Sample Loans")

        # Show a sample of the data
        display_df = df[['credit_score', 'annual_income', 'loan_amount', 'debt_to_income',
                        'predicted_probability', 'predicted_category', 'recommendation']].head(20)
        display_df['predicted_probability'] = display_df['predicted_probability'].apply(lambda x: f"{x:.1%}")

        st.dataframe(display_df, use_container_width=True)


def main():
    """Main application."""

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Score Individual Loan", "Model Performance", "Batch Analysis"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This dashboard demonstrates a machine learning system for
    predicting loan default risk.

    **Model**: XGBoost Classifier
    **Features**: 16+ borrower attributes
    **Accuracy**: ~85% on test data
    **ROC-AUC**: ~0.90
    """)

    # Route to appropriate page
    if page == "Home":
        home_page()
    elif page == "Score Individual Loan":
        score_loan_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Batch Analysis":
        batch_analysis_page()


if __name__ == "__main__":
    main()
