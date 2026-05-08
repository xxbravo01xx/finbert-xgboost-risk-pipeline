# HOW TO RUN:
#   1. Install dependencies:
#      pip install yfinance pandas numpy scikit-learn xgboost
#          transformers torch matplotlib seaborn statsmodels
#          requests beautifulsoup4 imbalanced-learn
#   2. Run the script top-to-bottom:
#      python ffmgf-risk-engine.py
#   3. Outputs:
#      - Console: model metrics, summaries
#      - Files: sentiment_data.csv, results_summary.csv,
#               figures/ directory with all plots


Overview
This repository contains a multimodal machine-learning pipeline developed for regulatory and sociopolitical risk forecasting in equities. Traditional deterministic financial models often fail to predict sudden price disruptions caused by qualitative exogenous shocks, such as environmental delays or community protests. Using First Mining Gold Corp. (FFMGF) as a case study, this project demonstrates how to quantify these unquantifiable risks to create an early-warning system for investors.  

The pipeline integrates Natural Language Processing (NLP) to extract structured sentiment signals from unstructured text corpora using FinBERT. These qualitative scores are combined with quantitative technical indicators to train ensemble classifiers (Random Forest, XGBoost) and deep learning networks (LSTM) to predict stock price crashes. 

Key FeaturesNLP Sentiment Extraction: Utilizes Hugging Face's FinBERT to map regulatory filings and news headlines to a numeric sentiment scale, capturing the aggregate regulatory environment.  Feature Engineering: Processes daily OHLCV data to engineer technical indicators including 20-day rolling volatility, RSI-14, MACD, and Bollinger Bandwidth.  

Crash Prediction Models: Compares the predictive accuracy of XGBoost, Random Forest, and LSTM architectures on imbalanced, time-series financial data.  

Anomaly Detection: Employs Z-score anomaly detection to isolate extreme return events and correlate them with macroeconomic or regulatory catalysts. 

Tech Stack
Language: Python   
Machine Learning: Scikit-learn, XGBoost, PyTorch, Imbalanced-learn (SMOTE)   
NLP: Hugging Face Transformers (FinBERT)   
Data Processing & APIs: Pandas, NumPy, yfinance 
