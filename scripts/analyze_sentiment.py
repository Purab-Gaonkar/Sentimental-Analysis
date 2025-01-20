from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('models/sentiment_model')
model = BertForSequenceClassification.from_pretrained('models/sentiment_model')
model.eval()  # Set the model to evaluation mode

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    sentiment = probs.argmax().item()
    return sentiment, probs[0][sentiment].item()

# Read new comments from a CSV file for analysis (modify as per your need)
comments_df = pd.read_csv("data/processed/sentiment_data.csv")

# Analyze sentiments of the comments
comments_df['sentiment'] = comments_df['comment'].apply(lambda x: analyze_sentiment(x)[0])
comments_df['probability'] = comments_df['comment'].apply(lambda x: analyze_sentiment(x)[1])

# Print the results
print(comments_df.head())

# Save the results to a new CSV file
comments_df.to_csv("data/processed/analyzed_sentiments.csv", index=False)
print("Sentiments analyzed and saved successfully in analyzed_sentiments.csv")
