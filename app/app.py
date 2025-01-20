from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('models/sentiment_model')
model = DistilBertForSequenceClassification.from_pretrained('models/sentiment_model')
model.eval()  # Set the model to evaluation mode

# Define sentiment labels (update according to your model's configuration)
sentiment_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

# Function to analyze sentiment
def analyze_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and apply softmax for probabilities
    probs = outputs.logits.softmax(dim=-1)
    
    # Get the sentiment (argmax to pick the highest probability)
    sentiment = probs.argmax().item()
    return sentiment, probs[0][sentiment].item()

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the data from the POST request
    data = request.json

    # Check if 'text' key exists in the received JSON data
    if 'text' not in data:
        return jsonify({'error': 'No text field found in the request'}), 400
    
    # Get the input text
    text = data['text']
    
    # Perform sentiment analysis
    sentiment, probability = analyze_sentiment(text)

    # Return the result
    return jsonify({
        'sentiment': sentiment_labels.get(sentiment, 'Unknown'),
        'probability': round(probability, 4)  # Round the probability to 4 decimal places
    })

if __name__ == '__main__':
    # Start Flask app (change host/port as needed)
    app.run(debug=True)
