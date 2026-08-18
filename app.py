"""
Streamlit Application for Telecom Customer Churn Prediction
Machine Learning Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef,
    ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

 
# PAGE CONFIG & STYLING
 

st.set_page_config(
    page_title="Churn Prediction | ML Assignment 2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern, clean CSS
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #ffffff; }
    
    /* Header styling */
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-box h2 { margin: 0; font-size: 28px; font-weight: 700; }
    .header-box p { margin: 5px 0 0 0; font-size: 14px; opacity: 0.95; }
    
    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #2c3e50;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 3px solid #667eea;
    }
    
    /* Card styling */
    .card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 15px 0;
    }
    
    /* Metric boxes */
    .metric-card {
        background-color: #f0f4ff;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e7ff;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
        margin: 10px 0 5px 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tables */
    .dataframe { border-collapse: collapse; }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

 
# LOAD MODELS AND PREPROCESSOR
 

@st.cache_resource
def load_models():
    """Load all trained models and preprocessor"""
    model_dir = Path("model")
    
    models = {}
    try:
        models['Logistic Regression'] = joblib.load(model_dir / 'logistic_regression_pipeline.pkl')
        models['Decision Tree'] = joblib.load(model_dir / 'decision_tree_pipeline.pkl')
        models['KNN'] = joblib.load(model_dir / 'knn_pipeline.pkl')
        models['Gaussian Naive Bayes'] = joblib.load(model_dir / 'gaussian_naive_bayes_pipeline.pkl')
        models['Random Forest'] = joblib.load(model_dir / 'random_forest_pipeline.pkl')
        return models, True
    except Exception as e:
        return {}, False

 
# EVALUATION FUNCTIONS
 

def evaluate_predictions(y_true, y_pred, y_proba, model_name="Model"):
    """Calculate all evaluation metrics"""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title("Confusion Matrix", fontweight='bold', fontsize=14)
    plt.tight_layout()
    return fig, cm

 
# MAIN APP
 

# Load models
models, models_loaded = load_models()

# Header
st.markdown("""
    <div class="header-box">
        <h2>🎓 ML Assignment 2: Churn Prediction</h2>
        <p><strong>Name:</strong> Avishkar Ghumare | <strong>BITS ID:</strong> 2025ac05448</p>
    </div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("❌ Models not found. Please ensure trained models are saved in the 'model/' directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        help="Choose which model to evaluate on test data"
    )
    
    st.markdown("---")
    
    st.markdown("### 📋 Available Models")
    for model in models.keys():
        if model == selected_model:
            st.markdown(f"✅ **{model}**")
        else:
            st.markdown(f"○ {model}")

# Main content
st.markdown("### 📊 Test Data & Model Evaluation")

# Upload section in a cleaner layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 📁 Upload Test Data")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", label_visibility="collapsed")
    test_data = None
    
    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            st.markdown(f"""
            <div class="info-box">
            ✅ Data loaded successfully<br>
            Samples: <strong>{test_data.shape[0]}</strong> | Features: <strong>{test_data.shape[1]}</strong>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        try:
            test_data = pd.read_csv(Path("data") / "test_data.csv")
            st.markdown(f"""
            <div class="info-box">
            📊 Using default test data<br>
            Samples: <strong>{test_data.shape[0]}</strong> | Features: <strong>{test_data.shape[1]}</strong>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.warning("No test data found. Please upload a CSV file.")

with col2:
    st.markdown("#### 🤖 Selected Model")
    st.markdown(f"""
    <div class="card">
    <strong>{selected_model}</strong><br>
    <small style="color: #666;">Ready for evaluation on test data</small>
    </div>
    """, unsafe_allow_html=True)

 
# PREDICTIONS AND ANALYSIS
 

if test_data is not None and 'Churn' in test_data.columns:
    # Separate features and target
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']
    
    # Make predictions
    pipeline = models[selected_model]
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = evaluate_predictions(y_test, y_pred, y_proba, selected_model)
    
     
    # SECTION 1: KEY METRICS
     
    
    st.markdown('<div class="section-header">📈 Performance Metrics</div>', unsafe_allow_html=True)
    
    # Enhanced CSS for metric cards
    st.markdown("""
    <style>
    .metric-card-enhanced {
        background: linear-gradient(135deg, var(--color-start) 0%, var(--color-end) 100%);
        padding: 22px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-enhanced:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .metric-card-enhanced::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    }
    
    .metric-icon-large {
        font-size: 32px;
        margin-bottom: 10px;
        display: block;
    }
    
    .metric-value-large {
        font-size: 36px;
        font-weight: 800;
        color: #1a1a1a;
        margin: 8px 0;
        text-shadow: 0 1px 2px rgba(255,255,255,0.3);
    }
    
    .metric-label-large {
        font-size: 13px;
        color: #333333;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .metric-bar {
        width: 100%;
        height: 6px;
        background-color: rgba(0,0,0,0.15);
        border-radius: 3px;
        margin-top: 12px;
        overflow: hidden;
    }
    
    .metric-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, rgba(100,100,100,0.8), #555555);
        border-radius: 3px;
        box-shadow: 0 0 8px rgba(100,100,100,0.5);
    }
    
    .metric-badge {
        display: inline-block;
        background-color: rgba(0,0,0,0.2);
        color: #222222;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        margin-top: 10px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Define metric configurations with colors
    metric_configs = [
        ('Accuracy', metrics['Accuracy'], '📊', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'),
        ('AUC', metrics['AUC'], '📈', 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'),
        ('Precision', metrics['Precision'], '🎯', 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'),
        ('Recall', metrics['Recall'], '🔍', 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'),
    ]
    
    metric_cols = st.columns(4, gap="medium")
    
    for col, (label, value, emoji, gradient) in zip(metric_cols, metric_configs):
        with col:
            # Determine quality badge
            if value >= 0.9:
                badge = "🌟 Excellent"
            elif value >= 0.8:
                badge = "✅ Very Good"
            elif value >= 0.7:
                badge = "👍 Good"
            else:
                badge = "⚠️ Fair"
            
            st.markdown(f"""
            <div class="metric-card-enhanced" style="--color-start: {gradient.split()[2]} 0%; --color-end: {gradient.split()[-2]} 100%;">
                <div class="metric-icon-large">{emoji}</div>
                <div class="metric-value-large">{value:.4f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {value * 100}%"></div>
                </div>
                <div class="metric-label-large">{label}</div>
                <div class="metric-badge">{badge}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")  # Spacing
    
    # Additional metrics row with enhanced styling
    col1, col2, col3 = st.columns(3, gap="medium")
    
    additional_metrics = [
        (col1, 'F1 Score', metrics['F1'], '⚡', 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'),
        (col2, 'MCC', metrics['MCC'], '🔗', 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)'),
        (col3, 'Test Samples', len(y_test), '📊', 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'),
    ]
    
    for col, label, value, emoji, gradient in additional_metrics:
        with col:
            if label == 'Test Samples':
                value_display = f"{int(value)}"
                badge = "📈 Dataset Size"
            else:
                value_display = f"{value:.4f}"
                if value >= 0.8:
                    badge = "✅ Strong"
                elif value >= 0.6:
                    badge = "👍 Decent"
                else:
                    badge = "⚠️ Needs Work"
            
            st.markdown(f"""
            <div class="metric-card-enhanced" style="--color-start: {gradient.split()[2]} 0%; --color-end: {gradient.split()[-2]} 100%;">
                <div class="metric-icon-large">{emoji}</div>
                <div class="metric-value-large">{value_display}</div>
                <div class="metric-label-large">{label}</div>
                <div class="metric-badge">{badge}</div>
            </div>
            """, unsafe_allow_html=True)
    
     
    # SECTION 2: CONFUSION MATRIX
     
    
    st.markdown('<div class="section-header">🔍 Confusion Matrix</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        fig_cm, cm = plot_confusion_matrix(y_test, y_pred)
        st.pyplot(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("#### Matrix Values")
        cm_data = {
            'TN (Correct Negatives)': cm[0, 0],
            'FP (False Positives)': cm[0, 1],
            'FN (False Negatives)': cm[1, 0],
            'TP (True Positives)': cm[1, 1],
        }
        for label, value in cm_data.items():
            st.markdown(f"**{label}:** `{value}`")
    
     
    # SECTION 3: CLASSIFICATION REPORT
     
    
    st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)
    
    # Get classification report as dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Create a formatted dataframe from the report
    report_data = []
    
    # Add data for each class (0 and 1)
    for class_label in ['0', '1']:
        class_name = 'No Churn' if class_label == '0' else 'Churn'
        if class_label in report_dict:
            report_data.append({
                'Class': class_name,
                'Precision': f"{report_dict[class_label]['precision']:.4f}",
                'Recall': f"{report_dict[class_label]['recall']:.4f}",
                'F1-Score': f"{report_dict[class_label]['f1-score']:.4f}",
                'Support': int(report_dict[class_label]['support'])
            })
    
    # Add macro and weighted averages
    report_data.append({
        'Class': '🔹 Macro Avg',
        'Precision': f"{report_dict['macro avg']['precision']:.4f}",
        'Recall': f"{report_dict['macro avg']['recall']:.4f}",
        'F1-Score': f"{report_dict['macro avg']['f1-score']:.4f}",
        'Support': int(report_dict['macro avg']['support'])
    })
    
    report_data.append({
        'Class': '🔹 Weighted Avg',
        'Precision': f"{report_dict['weighted avg']['precision']:.4f}",
        'Recall': f"{report_dict['weighted avg']['recall']:.4f}",
        'F1-Score': f"{report_dict['weighted avg']['f1-score']:.4f}",
        'Support': int(report_dict['weighted avg']['support'])
    })
    
    report_df = pd.DataFrame(report_data)
    
    # Display as styled table
    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Add interpretation guide
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Metric Definitions:**
        - **Precision**: Of predicted positives, how many are actually positive?
        - **Recall**: Of actual positives, how many did we predict?
        - **F1-Score**: Harmonic mean of precision and recall
        - **Support**: Number of samples in that class
        """)
    
    with col2:
        st.markdown(f"""
        **📈 Model Performance Summary:**
        - **No Churn Class**: {int(report_dict['0']['support'])} samples
        - **Churn Class**: {int(report_dict['1']['support'])} samples
        - **Weighted F1**: {report_dict['weighted avg']['f1-score']:.4f}
        - **Overall Accuracy**: {metrics['Accuracy']:.4f}
        """)
    
     
    # SECTION 4: MODEL COMPARISON
     
    
    st.markdown('<div class="section-header">🏆 All Models Comparison</div>', unsafe_allow_html=True)
    
    comparison_data = []
    for model_name, model in models.items():
        y_pred_model = model.predict(X_test)
        y_proba_model = model.predict_proba(X_test)[:, 1]
        model_metrics = evaluate_predictions(y_test, y_pred_model, y_proba_model, model_name)
        comparison_data.append(model_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format and display
    comparison_display = comparison_df.copy()
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        comparison_display[col] = comparison_display[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(comparison_display, use_container_width=True, hide_index=True)
    
    # Highlight best model
    best_model_idx = comparison_df['AUC'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_auc = comparison_df.loc[best_model_idx, 'AUC']
    
    st.markdown(f"""
    <div class="success-box">
    <strong>🏆 Best Model (by AUC):</strong> <br>
    <strong style="font-size: 18px;">{best_model_name}</strong> | AUC: <strong>{best_auc:.4f}</strong>
    </div>
    """, unsafe_allow_html=True)
    
     
    # SECTION 5: PREDICTION SAMPLES
     
    
    st.markdown('<div class="section-header">🎯 Sample Predictions</div>', unsafe_allow_html=True)
    
    results_df = X_test.copy()
    results_df['Actual'] = y_test.values
    results_df['Predicted'] = y_pred
    results_df['Probability'] = y_proba.round(4)
    results_df['Status'] = results_df.apply(
        lambda x: '✅' if x['Actual'] == x['Predicted'] else '❌', 
        axis=1
    )
    
    display_df = results_df[['Actual', 'Predicted', 'Probability', 'Status']].head(10)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Prediction statistics
    col1, col2, col3 = st.columns(3)
    correct_predictions = (y_test == y_pred).sum()
    incorrect_predictions = (y_test != y_pred).sum()
    
    with col1:
        st.metric("✅ Correct Predictions", correct_predictions)
    with col2:
        st.metric("❌ Incorrect Predictions", incorrect_predictions)
    with col3:
        accuracy_pct = (correct_predictions / len(y_test)) * 100
        st.metric("Accuracy %", f"{accuracy_pct:.1f}%")

else:
    st.markdown("""
    <div class="info-box">
    <strong>⚠️ No Valid Data</strong><br>
    Please upload a CSV file with a 'Churn' column, or ensure test data exists at <code>data/test_data.csv</code>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
    📊 Telecom Customer Churn Prediction | ML Assignment 2<br>
    Avishkar Ghumare (2025ac05448) | 2026
    </div>
""", unsafe_allow_html=True)
