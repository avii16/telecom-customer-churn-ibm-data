"""
Gaussian Naive Bayes Classifier Model
Module for implementing and training Gaussian Naive Bayes classifier.

Important: GaussianNB requires dense input. The preprocessing pipeline
converts sparse matrices to dense format appropriately.
"""

from sklearn.naive_bayes import GaussianNB
import joblib


def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes Classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (must be dense)
    y_train : array-like
        Training target variable
    
    Returns:
    --------
    model : GaussianNB
        Trained Gaussian Naive Bayes model
    """
    # Convert sparse matrix to dense if necessary
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict_naive_bayes(model, X_test):
    """
    Make predictions using trained Gaussian Naive Bayes model.
    
    Parameters:
    -----------
    model : GaussianNB
        Trained model
    X_test : array-like
        Test features (must be dense)
    
    Returns:
    --------
    y_pred : array
        Predicted class labels
    y_proba : array
        Prediction probabilities
    """
    # Convert sparse matrix to dense if necessary
    if hasattr(X_test, 'toarray'):
        X_test = X_test.toarray()
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    return y_pred, y_proba


def save_model(model, filepath):
    """Save model to disk using joblib."""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model from disk using joblib."""
    return joblib.load(filepath)
