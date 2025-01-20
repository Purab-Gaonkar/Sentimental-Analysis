# preprocess_data.py
import csv
from transformers import BertTokenizer
import emoji

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')

# Read comments from the file
with open("data/raw/ytcomments.txt", 'r', encoding='utf-8') as f:
    comments = f.readlines()

# Function to preprocess the data
def preprocess(comment):
    comment = comment.lower().strip()
    emojis = emoji.emoji_count(comment)
    text_characters = len(comment.replace(" ", ""))
    if emojis == 0 or (text_characters / (text_characters + emojis)) > 0.65:
        return comment
    return None

# Preprocess comments and write to CSV
with open("data/processed/sentiment_data.csv", 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['comment']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for comment in comments:
        processed_comment = preprocess(comment)
        if processed_comment:
            writer.writerow({'comment': processed_comment})

print("Sentiment data stored in sentiment_data.csv")
