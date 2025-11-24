"""
Text preprocessing and feature extraction for sentiment analysis.
"""
import re
import string
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Handle contractions (basic)
    contractions = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text and optionally remove stopwords.
    
    Args:
        text: Cleaned text string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of tokens
    """
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Remove single character tokens
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens


def preprocess_reviews(df: pd.DataFrame, text_column: str = 'review') -> pd.DataFrame:
    """
    Apply preprocessing pipeline to DataFrame of reviews.
    
    Args:
        df: DataFrame containing reviews
        text_column: Name of column containing text
        
    Returns:
        DataFrame with additional 'cleaned_text' and 'tokens' columns
    """
    df = df.copy()
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Tokenize
    print("Tokenizing...")
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    
    # Join tokens back for TF-IDF
    df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    return df


def create_tfidf_features(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.8
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF feature matrix from texts.
    
    Args:
        texts: List of preprocessed text strings
        max_features: Maximum number of features
        ngram_range: Range of n-grams to extract
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        
    Returns:
        Tuple of (feature matrix, fitted vectorizer)
    """
    print(f"Creating TF-IDF features with max_features={max_features}, ngram_range={ngram_range}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True  # Use log scaling for term frequency
    )
    
    X = vectorizer.fit_transform(texts)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X, vectorizer


def encode_labels(labels: pd.Series) -> np.ndarray:
    """
    Encode sentiment labels to binary (0/1).
    
    Args:
        labels: Series of sentiment labels ('positive'/'negative')
        
    Returns:
        Binary encoded labels (1 for positive, 0 for negative)
    """
    return (labels == 'positive').astype(int).values
