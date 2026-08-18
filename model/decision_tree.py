"""
Decision Tree Classifier Model
Module for implementing and training decision tree classifier.
"""

from sklearn.tree import DecisionTreeClassifier
import joblib


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a Decision Tree Classifier.
    
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
    model : DecisionTreeClassifier
        Trained decision tree model
    """
    model = DecisionTreeClassifier(random_state=random_state, max_depth=10)
    model.fit(X_train, y_train)
    return model


def predict_decision_tree(model, X_test):
    """
    Make predictions using trained Decision Tree model.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
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
