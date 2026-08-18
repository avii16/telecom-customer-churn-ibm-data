"""
Random Forest Classifier Model
Module for implementing and training random forest classifier.

Random Forest is an ensemble method that combines multiple decision trees
to improve prediction accuracy and reduce overfitting.
"""

from sklearn.ensemble import RandomForestClassifier
import joblib


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target variable
    n_estimators : int
        Number of trees in the forest
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    model : RandomForestClassifier
        Trained random forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Use all available processors
    )
    model.fit(X_train, y_train)
    return model


def predict_random_forest(model, X_test):
    """
    Make predictions using trained Random Forest model.
    
    Parameters:
    -----------
    model : RandomForestClassifier
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


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    feature_names : list
        List of feature names
    
    Returns:
    --------
    importance_dict : dict
        Dictionary mapping feature names to importance scores
    """
    importances = model.feature_importances_
    return {name: imp for name, imp in zip(feature_names, importances)}


def save_model(model, filepath):
    """Save model to disk using joblib."""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model from disk using joblib."""
    return joblib.load(filepath)
