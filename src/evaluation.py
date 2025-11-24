"""
Model evaluation and visualization utilities.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.base import BaseEstimator


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with predictions, probabilities, and metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'metrics': metrics,
        'classification_report': report
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of model for title
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create annotations with counts and percentages
    annotations = np.array([[f'{count}\n({percent:.1f}%)' 
                            for count, percent in zip(row_counts, row_percents)]
                           for row_counts, row_percents in zip(cm, cm_percent)])
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def extract_top_features(
    model: BaseEstimator,
    vectorizer,
    n_features: int = 20
) -> pd.DataFrame:
    """
    Extract top features by importance/weight.
    
    Args:
        model: Trained model (Naive Bayes or Logistic Regression)
        vectorizer: Fitted TF-IDF vectorizer
        n_features: Number of top features to extract
        
    Returns:
        DataFrame with features and their weights
    """
    feature_names = vectorizer.get_feature_names_out()
    
    # Get feature weights based on model type
    if hasattr(model, 'feature_log_prob_'):
        # Naive Bayes
        # Get log probabilities for positive class
        log_probs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        weights = log_probs
    elif hasattr(model, 'coef_'):
        # Logistic Regression
        weights = model.coef_[0]
    else:
        raise ValueError("Model type not supported")
    
    # Create DataFrame
    features_df = pd.DataFrame({
        'feature': feature_names,
        'weight': weights
    })
    
    # Sort by absolute weight
    features_df['abs_weight'] = features_df['weight'].abs()
    features_df = features_df.sort_values('abs_weight', ascending=False)
    
    # Get top positive and negative features
    top_positive = features_df.nlargest(n_features, 'weight')
    top_negative = features_df.nsmallest(n_features, 'weight')
    
    return pd.concat([top_positive, top_negative]).drop('abs_weight', axis=1)


def plot_feature_importance(
    features_df: pd.DataFrame,
    model_name: str,
    save_path: str = None,
    n_features: int = 20
) -> None:
    """
    Plot feature importance bar chart.
    
    Args:
        features_df: DataFrame with features and weights
        model_name: Name of model for title
        save_path: Path to save figure (optional)
        n_features: Number of features to display per side
    """
    # Separate positive and negative features
    positive_features = features_df[features_df['weight'] > 0].nlargest(n_features, 'weight').sort_values('weight')
    negative_features = features_df[features_df['weight'] < 0].nsmallest(n_features, 'weight').sort_values('weight')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot positive features
    if len(positive_features) > 0:
        ax1.barh(positive_features['feature'], positive_features['weight'], color='green', alpha=0.7)
        ax1.set_xlabel('Weight', fontsize=12)
        ax1.set_title(f'Top {len(positive_features)} Positive Features', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No positive features', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Top Positive Features', fontsize=12, fontweight='bold')
    
    # Plot negative features
    if len(negative_features) > 0:
        ax2.barh(negative_features['feature'], negative_features['weight'], color='red', alpha=0.7)
        ax2.set_xlabel('Weight', fontsize=12)
        ax2.set_title(f'Top {len(negative_features)} Negative Features', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No negative features', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Top Negative Features', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_roc_curves(
    models_dict: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None
) -> None:
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_dict: Dictionary of model_name: model
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models_dict.items():
        # Get probability predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def compare_models(
    results_dict: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create comparison table of model metrics.
    
    Args:
        results_dict: Dictionary of model_name: evaluation results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = {}
    
    for model_name, results in results_dict.items():
        comparison_data[model_name] = results['metrics']
    
    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df = comparison_df.round(4)
    
    return comparison_df
