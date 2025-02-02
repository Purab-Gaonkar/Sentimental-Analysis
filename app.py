from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from flask_cors import CORS
import re
import requests
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load DistilBERT model and tokenizer (from your trained model)
model_name = "models/sentiment_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

# Ensure the code runs on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the appropriate device (GPU/CPU)
model = model.to(device)

# YouTube API key (replace with your own)
YOUTUBE_API_KEY = "AIzaSyBtJjniAVMDaQN3OaocZpP7MyAVBgXPl2Q"

# Sentiment labels
sentiment_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

# Function to analyze sentiment using the trained model
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure inputs are moved to the correct device
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and apply softmax for probabilities
    probs = outputs.logits.softmax(dim=-1)
    sentiment = probs.argmax().item()  # Get the index of the highest probability
    confidence = round(probs[0][sentiment].item(), 4)
    
    return sentiment_labels.get(sentiment, 'Unknown'), confidence

# Extract video ID from YouTube URL
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

# Fetch YouTube comments
def get_youtube_comments(video_id):
    url = f"https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": YOUTUBE_API_KEY,
        "maxResults": 100,  # Adjust number of comments to retrieve
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    for item in response.json().get("items", [])]
        return comments
    return []

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    # Extract video ID
    video_id = extract_video_id(data['url'])
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    # Fetch comments
    comments = get_youtube_comments(video_id)
    if not comments:
        return jsonify({'error': 'No comments found'}), 404

    # Analyze sentiments
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    analyzed_comments = []

    for comment in comments:
        sentiment, prob = analyze_sentiment(comment)
        sentiment_counts[sentiment] += 1
        analyzed_comments.append({
            "comment": comment,
            "sentiment": sentiment,
            "probability": prob
        })

    # Determine overall sentiment based on the highest count of sentiments
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return jsonify({
        "videoId": video_id,
        "overall_sentiment": overall_sentiment,  # Added overall sentiment of the video
        "sentiment_summary": sentiment_counts,
        "analyzed_comments": analyzed_comments  # Clean output for each comment
    })

if __name__ == '__main__':
    app.run(debug=True)
