# IMDB Movie Review Sentiment Analysis

A comprehensive sentiment analysis project comparing Naive Bayes and Logistic Regression classifiers on the IMDB movie reviews dataset (50,000 reviews).

## 🎯 Project Overview

This project implements and compares two machine learning approaches for binary sentiment classification:
- **Naive Bayes (MultinomialNB)**: Fast, probabilistic baseline
- **Logistic Regression**: More sophisticated linear classifier

### Key Features
- Complete preprocessing pipeline (cleaning, tokenization, TF-IDF)
- Comprehensive model evaluation and comparison
- Interactive prediction interface
- Feature importance analysis
- Detailed visualizations and metrics

## 📊 Dataset

**IMDB Movie Reviews Dataset**
- **Size**: 50,000 reviews
- **Balance**: 25,000 positive, 25,000 negative
- **Source**: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Format**: CSV with review text and sentiment labels

## 🏗️ Project Structure

```
cs410-project/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── run_pipeline.py                    # Automated pipeline execution
├── setup.py                           # Environment setup verification
├── data/
│   └── IMDB Dataset.csv              # Raw dataset (50K reviews)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and data analysis
│   ├── 02_preprocessing.ipynb        # Text preprocessing
│   ├── 03_model_training.ipynb       # Model training
│   ├── 04_evaluation_comparison.ipynb # Evaluation & comparison
│   └── 05_interactive_predictor.ipynb # Interactive interface
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Text preprocessing (171 lines)
│   ├── models.py                     # Model training (110 lines)
│   ├── evaluation.py                 # Evaluation utilities (283 lines)
│   └── utils.py                      # Helper functions (113 lines)
├── models/
│   ├── naive_bayes_model.pkl         # Trained NB model
│   ├── logistic_regression_model.pkl # Trained LR model
│   ├── tfidf_vectorizer.pkl          # Fitted vectorizer
│   └── train_test_data.npz           # Preprocessed data
└── results/
    ├── metrics_comparison.csv        # Performance metrics
    ├── metrics_comparison.png        # Metrics bar chart
    ├── roc_curves.png                # ROC curves comparison
    ├── confidence_distributions.png  # Confidence analysis
    ├── prediction_agreement.png      # Model agreement heatmap
    ├── sentiment_analysis_presentation.pptx  # PowerPoint presentation
    ├── confusion_matrices/           # Confusion matrix plots
    │   ├── naive_bayes_cm.png
    │   └── logistic_regression_cm.png
    └── feature_importance/           # Feature analysis plots
        ├── naive_bayes_features.png
        └── logistic_regression_features.png
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cs410-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

4. **Verify setup**
```bash
python setup.py
```

5. **Verify data**
Ensure `data/IMDB Dataset.csv` exists. If not, download from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Quick Start

Run the entire pipeline automatically:
```bash
python run_pipeline.py
```

This executes all notebooks in sequence and generates all results.

## 📖 Usage Guide

### Running Notebooks

Execute notebooks in order:

1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Load and explore dataset
   - Analyze text statistics
   - Visualize sentiment distribution

2. **Preprocessing** (`02_preprocessing.ipynb`)
   - Clean and tokenize text
   - Create TF-IDF features
   - Split train/test sets

3. **Model Training** (`03_model_training.ipynb`)
   - Train Naive Bayes classifier
   - Train Logistic Regression classifier
   - Save trained models

4. **Evaluation & Comparison** (`04_evaluation_comparison.ipynb`)
   - Calculate performance metrics
   - Generate confusion matrices
   - Analyze feature importance
   - Compare models

5. **Interactive Predictor** (`05_interactive_predictor.ipynb`)
   - Use interactive widget for predictions
   - Test with custom reviews
   - Batch prediction

### Using the Interactive Predictor

```python
# In Jupyter notebook
from src.models import load_model
from src.preprocessing import clean_text, tokenize_text
import pickle

# Load models
nb_model = load_model('models/naive_bayes_model.pkl')
lr_model = load_model('models/logistic_regression_model.pkl')

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict sentiment
review = "This movie was absolutely fantastic!"
cleaned = clean_text(review)
tokens = tokenize_text(cleaned)
X = vectorizer.transform([' '.join(tokens)])

nb_prediction = nb_model.predict(X)[0]
lr_prediction = lr_model.predict(X)[0]

print(f"Naive Bayes: {'Positive' if nb_prediction == 1 else 'Negative'}")
print(f"Logistic Regression: {'Positive' if lr_prediction == 1 else 'Negative'}")
```

## 🔬 Technical Details

### Preprocessing Pipeline

1. **Text Cleaning**
   - Convert to lowercase
   - Remove HTML tags, URLs, emails
   - Handle contractions
   - Remove special characters and digits
   - Remove extra whitespace

2. **Tokenization**
   - Word-level tokenization
   - Stopword removal (NLTK English)
   - Filter single-character tokens

3. **Feature Extraction**
   - TF-IDF vectorization
   - Max features: 5,000
   - N-gram range: (1, 2) - unigrams and bigrams
   - Min document frequency: 5
   - Max document frequency: 0.8

### Model Configurations

**Naive Bayes (MultinomialNB)**
- Alpha: 1.0 (Laplace smoothing)
- Suitable for discrete features (word counts)
- Fast training and prediction

**Logistic Regression**
- C: 1.0 (inverse regularization strength)
- Max iterations: 1000
- Solver: liblinear
- Random state: 42

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction reliability
- **Recall**: Positive case coverage
- **F1-Score**: Harmonic mean of precision/recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: True/false positives/negatives

## 📈 Results Summary

### Model Performance

| Metric | Naive Bayes | Logistic Regression |
|--------|-------------|---------------------|
| Accuracy | 86.44% | 89.87% |
| Precision | 85.46% | 89.13% |
| Recall | 87.82% | 90.82% |
| F1-Score | 86.62% | 89.97% |
| ROC-AUC | 0.9344 | 0.9596 |
| Training Time | ~2-3 sec | ~10-15 sec |

### Key Findings

1. **Performance**: Logistic Regression outperforms Naive Bayes by 3.4% accuracy
2. **Speed**: Naive Bayes trains ~5x faster than Logistic Regression
3. **Agreement**: Models agree on 93.5% of predictions
4. **Confidence**: Logistic Regression has better calibrated confidence scores
5. **ROC-AUC**: Both models show excellent discrimination (>0.93)

### Feature Insights

**Top Positive Indicators**:
- excellent, great, perfect, amazing, wonderful
- best, loved, brilliant, fantastic, superb

**Top Negative Indicators**:
- worst, terrible, awful, bad, boring
- waste, poor, disappointing, horrible, dull

**Important Bigrams**:
- "highly recommend", "must see", "waste time"
- "not good", "don't waste", "well done"

## 🎨 Visualizations

The project generates comprehensive visualizations in `results/`:

1. **Confusion Matrices** (`confusion_matrices/`)
   - Naive Bayes: 86.44% accuracy breakdown
   - Logistic Regression: 89.87% accuracy breakdown

2. **ROC Curves** (`roc_curves.png`)
   - Comparative ROC curves with AUC scores
   - NB: 0.9344 AUC, LR: 0.9596 AUC

3. **Feature Importance** (`feature_importance/`)
   - Top 20 positive and negative features per model
   - Interpretable sentiment indicators

4. **Metrics Comparison** (`metrics_comparison.png`)
   - Side-by-side bar chart of all metrics
   - Clear performance comparison

5. **Confidence Distributions** (`confidence_distributions.png`)
   - Model prediction confidence analysis
   - Calibration comparison

6. **Prediction Agreement** (`prediction_agreement.png`)
   - Heatmap showing where models agree/disagree
   - 93.5% agreement rate

7. **PowerPoint Presentation** (`sentiment_analysis_presentation.pptx`)
   - 14 professional slides with all results
   - Ready for course presentation

## 🔮 Future Enhancements

Potential improvements (out of current scope):

- Deep learning models (LSTM, BERT, Transformers)
- Multi-class sentiment (positive/neutral/negative)
- Aspect-based sentiment analysis
- Real-time streaming predictions
- Web application deployment
- Cross-validation and hyperparameter tuning
- Ensemble methods
- Explainable AI techniques (LIME, SHAP)

## 🙏 Acknowledgments

- IMDB dataset from Kaggle
- scikit-learn documentation and examples
- NLTK library for NLP preprocessing
- Course materials and instructors
