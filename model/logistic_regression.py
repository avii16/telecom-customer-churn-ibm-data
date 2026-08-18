"""
Logistic Regression Model
Module for implementing and training logistic regression classifier.
"""

from sklearn.linear_model import LogisticRegression
import joblib


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target variable
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    model : LogisticRegression
        Trained logistic regression model
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_logistic_regression(model, X_test):
    """
    Make predictions using trained Logistic Regression model.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X_test : array-like
        Test features
    
    Returns:
    --------
    y_pred : array
        Predicted class labels
    y_proba : array
        Prediction probabilities
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    return y_pred, y_proba


def save_model(model, filepath):
    """Save model to disk using joblib."""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model from disk using joblib."""
    return joblib.load(filepath)
