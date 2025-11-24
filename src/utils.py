"""
Utility functions for the sentiment analysis project.
"""
import os
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_imdb_data(filepath: str = 'data/IMDB Dataset.csv') -> pd.DataFrame:
    """
    Load IMDB dataset from CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with reviews and sentiments
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {df.columns.tolist()}")
    return df


def split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix (can be sparse or dense)
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train positive ratio: {np.sum(y_train) / len(y_train):.2%}")
    print(f"Test positive ratio: {np.sum(y_test) / len(y_test):.2%}")
    
    return X_train, X_test, y_train, y_test


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def print_data_stats(df: pd.DataFrame, text_column: str = 'review') -> None:
    """
    Print statistics about the dataset.
    
    Args:
        df: DataFrame with reviews
        text_column: Name of text column
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal reviews: {len(df):,}")
    
    if 'sentiment' in df.columns:
        print("\nSentiment distribution:")
        print(df['sentiment'].value_counts())
        print(f"\nPositive ratio: {(df['sentiment'] == 'positive').mean():.2%}")
    
    if text_column in df.columns:
        df['text_length'] = df[text_column].str.len()
        df['word_count'] = df[text_column].str.split().str.len()
        
        print(f"\nText length statistics:")
        print(f"  Mean: {df['text_length'].mean():.0f} characters")
        print(f"  Median: {df['text_length'].median():.0f} characters")
        print(f"  Min: {df['text_length'].min()} characters")
        print(f"  Max: {df['text_length'].max()} characters")
        
        print(f"\nWord count statistics:")
        print(f"  Mean: {df['word_count'].mean():.0f} words")
        print(f"  Median: {df['word_count'].median():.0f} words")
        print(f"  Min: {df['word_count'].min()} words")
        print(f"  Max: {df['word_count'].max()} words")
    
    print("\n" + "="*50 + "\n")
