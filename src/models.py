"""
Machine learning models for sentiment classification.
"""
import pickle
from typing import Any, Dict
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator


def train_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> MultinomialNB:
    """
    Train Multinomial Naive Bayes classifier.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        alpha: Laplace smoothing parameter
        
    Returns:
        Trained Naive Bayes model
    """
    print("Training Naive Bayes classifier...")
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    print("Training complete!")
    return model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train Logistic Regression classifier.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        C: Inverse regularization strength
        max_iter: Maximum iterations
        random_state: Random seed
        
    Returns:
        Trained Logistic Regression model
    """
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver='liblinear',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Training complete!")
    return model


def save_model(model: BaseEstimator, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained scikit-learn model
        filepath: Path to save model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> BaseEstimator:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def predict_with_confidence(model: BaseEstimator, X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Make predictions with confidence scores.
    
    Args:
        model: Trained model
        X: Feature matrix
        
    Returns:
        Dictionary with predictions and probabilities
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'confidence': np.max(probabilities, axis=1)
    }
