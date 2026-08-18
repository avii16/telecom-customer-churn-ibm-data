"""
K-Nearest Neighbors Classifier Model
Module for implementing and training KNN classifier.

KNN requires scaled features because it relies on distance metrics.
Without scaling, features with larger ranges dominate the distance calculation.
"""

from sklearn.neighbors import KNeighborsClassifier
import joblib


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors Classifier.
    
    Note: Preprocessing pipeline handles feature scaling (StandardScaler).
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should already be scaled)
    y_train : array-like
        Training target variable
    n_neighbors : int
        Number of neighbors to use
    
    Returns:
    --------
    model : KNeighborsClassifier
        Trained KNN model
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def predict_knn(model, X_test):
    """
    Make predictions using trained KNN model.
    
    Parameters:
    -----------
    model : KNeighborsClassifier
        Trained model
    X_test : array-like
        Test features (should already be scaled)
    
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
