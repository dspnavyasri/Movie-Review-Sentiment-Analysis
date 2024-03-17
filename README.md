# Movie Review Sentiment Analysis

## Overview
This project aims to perform sentiment analysis on movie reviews using various machine learning classifiers. The sentiment analysis involves predicting whether a movie review expresses a positive or negative sentiment based on its text.

## Data
The dataset used in this project consists of movie reviews with corresponding sentiment labels. The data has been preprocessed to remove non-alphabetic characters, convert text to lowercase, tokenize words, remove stopwords, and lemmatize tokens. The final dataset has been balanced using oversampling techniques.

## Dependencies
- Python 3.x
- pandas
- nltk
- scikit-learn
- imbalanced-learn
- textblob

Install dependencies using:
```bash
pip install pandas nltk scikit-learn imbalanced-learn textblob
```

## Workflow
1. **Data Preprocessing**: The raw movie review data is cleaned, tokenized, and preprocessed to prepare it for analysis.
2. **Sentiment Analysis**: Sentiment analysis is performed using TextBlob to assign sentiment labels (positive or negative) to each review.
3. **Balancing Classes**: The dataset is balanced using oversampling techniques to address class imbalance.
4. **Model Training and Evaluation**: Several machine learning classifiers are trained and evaluated on the balanced dataset. Classifiers include Naive Bayes, Support Vector Machine (SVM), Random Forest, Logistic Regression, and Gradient Boosting.
5. **Hyperparameter Tuning**: Hyperparameter tuning is performed on selected models to optimize their performance.

## Results
- **Naive Bayes**: Achieved an accuracy of 83% with satisfactory precision, recall, and F1-score.
- **SVM**: Provided competitive performance with an accuracy of 83% and balanced precision and recall.
- **Random Forest**: Outperformed other models with an accuracy of 95% and high precision, recall, and F1-score.
- **Logistic Regression**: Demonstrated balanced performance with an accuracy of 94% and high precision, recall, and F1-score.
- **Gradient Boosting**: Achieved an accuracy of 84% with balanced precision, recall, and F1-score.

## Usage
1. Clone this repository:
```bash
git clone <repository_url>
```
2. Install dependencies.
3. Run the Jupyter notebook or Python script to perform sentiment analysis on your movie reviews.

## Future Improvements
- Explore advanced feature engineering techniques.
- Experiment with deep learning models for sentiment analysis.
- Fine-tune hyperparameters further for improved performance.
