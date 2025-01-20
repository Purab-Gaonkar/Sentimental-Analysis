# train_model.py
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric, Dataset
import numpy as np

# Load pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')
model = BertForSequenceClassification.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')

# Function to preprocess the data
def preprocess_function(examples):
    return tokenizer(examples['comment'], truncation=True, padding=True)

# Read comments from the file
dataset = Dataset.from_csv("data/processed/sentiment_data.csv")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Load metric
metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained('models/sentiment_model')
tokenizer.save_pretrained('models/sentiment_model')

print("Model trained and saved successfully!")
