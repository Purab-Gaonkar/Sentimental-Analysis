# app.py
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('models/sentiment_model')
model = BertForSequenceClassification.from_pretrained('models/sentiment_model')
model.eval()  # Set the model to evaluation mode

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    sentiment, probability = analyze_sentiment(text)
    return jsonify({'sentiment': sentiment, 'probability': probability})

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    sentiment = probs.argmax().item()
    return sentiment, probs[0][sentiment].item()

if __name__ == '__main__':
    app.run(debug=True)
