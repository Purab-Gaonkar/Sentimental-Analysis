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


## Setup Instructions

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd sentiment_analysis_project
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Collect YouTube comments:**
    ```sh
    python scripts/collect_data.py
    ```

4. **Preprocess data:**
    ```sh
    python scripts/preprocess_data.py
    ```

5. **Train the model:**
    ```sh
    python scripts/train_model.py
    ```

6. **Run the Flask app:**
    ```sh
    python app/app.py
    ```

## Usage

- **Collect YouTube comments using `collect_data.py`:**
    ```sh
    python scripts/collect_data.py
    ```

- **Preprocess comments data using `preprocess_data.py`:**
    ```sh
    python scripts/preprocess_data.py
    ```

- **Train the sentiment analysis model using `train_model.py`:**
    ```sh
    python scripts/train_model.py
    ```

- **Deploy the model as an API using `app.py`:**
    ```sh
    python app/app.py
    ```

## API Endpoint

- **`/analyze` (POST):** Analyze sentiment of the given text.
    - Request body: `{ "text": "Sample comment" }`
    - Response: `{ "sentiment": 2, "probability": 0.98 }`
