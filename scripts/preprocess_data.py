import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import emoji

# Download the VADER lexicon from NLTK
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer from VADER
sia = SentimentIntensityAnalyzer()

# Function to preprocess the data
def preprocess(comment):
    comment = comment.lower().strip()
    emojis_count = emoji.emoji_count(comment)
    text_characters = len(comment.replace(" ", ""))
    
    # Keep only the comments that contain more text than emojis
    if emojis_count == 0 or (text_characters / (text_characters + emojis_count)) > 0.65:
        return comment
    return None

# Function to classify sentiment using VADER lexicon and return numeric labels
def lexicon_based_sentiment(comment):
    # Get sentiment scores using VADER
    sentiment_score = sia.polarity_scores(comment)
    
    # Classify sentiment based on compound score and return numeric labels
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return 1  # Positive sentiment
    elif compound_score <= -0.05:
        return 0  # Negative sentiment
    else:
        return 2  # Neutral sentiment

# Read comments from the file
with open("data/raw/ytcomments.txt", 'r', encoding='utf-8') as f:
    comments = f.readlines()

# Preprocess comments and write to CSV
with open("data/processed/sentiment_data.csv", 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['comment', 'sentiment_label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for comment in comments:
        processed_comment = preprocess(comment)
        if processed_comment:
            sentiment_label = lexicon_based_sentiment(processed_comment)  # Get sentiment using lexicon
            writer.writerow({'comment': processed_comment, 'sentiment_label': sentiment_label})

print("Sentiment data stored in sentiment_data.csv")
