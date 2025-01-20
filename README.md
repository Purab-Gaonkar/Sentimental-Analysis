# Sentiment Analysis Project

This project involves collecting YouTube comments, training a sentiment analysis model, and deploying the model using a Flask API.

## Project Structure

sentiment_analysis_project/
├── data/
│   ├── raw/
│   │   └── ytcomments.txt         # Raw YouTube comments data
│   └── processed/
│       └── sentiment_data.csv     # Processed sentiment data
│       └── analyzed_sentiments.csv # Analyzed sentiment data with predictions
├── models/
│   └── sentiment_model/           # Directory to save trained model and tokenizer
├── scripts/
│   ├── collect_data.py            # Script to collect YouTube comments
│   ├── preprocess_data.py         # Script to preprocess data
│   ├── train_model.py             # Script to train the sentiment analysis model
│   └── analyze_sentiment.py       # Script to analyze sentiments using the trained model
├── app/
│   └── app.py                     # Flask app for deploying the model
├── requirements.txt               # List of project dependencies
└── README.md                      # Project documentation
