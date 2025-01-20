import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load
import numpy as np
import os
import pandas as pd

# Define model and tokenizer path
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
cache_dir = "./cache"  # Local cache directory for Hugging Face models

# Ensure the code runs on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")
else:
    print("CUDA is available. Using GPU.")

try:
    # Load pre-trained tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Switching to an alternative model...")
    # Fallback to a default model if the target model is unavailable
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6, cache_dir=cache_dir)  # Adjust num_labels as needed

# Move model to GPU if available
model = model.to(device)

# Function to preprocess the data
def preprocess_function(examples):
    # Here, we assume that 'label' is already present in the dataset
    return tokenizer(examples['comment'], truncation=True, padding=True)

# Read comments from the file
if os.path.exists("data/processed/sentiment_data.csv"):
    dataset = pd.read_csv("data/processed/sentiment_data.csv")

    # Check if 'sentiment_label' column exists, rename to 'labels' if it does
    if 'sentiment_label' in dataset.columns:
        dataset = dataset.rename(columns={"sentiment_label": "labels"})
    else:
        raise ValueError("The dataset does not have 'sentiment_label' column. Ensure your data is labeled.")

    # Convert the pandas DataFrame to a Hugging Face Dataset object
    dataset = Dataset.from_pandas(dataset)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
else:
    raise FileNotFoundError("Dataset file not found: data/processed/sentiment_data.csv")

# Split dataset into training and evaluation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',  # Evaluate after each epoch
    save_strategy='epoch',       # Save the model at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,  # Load the best model at the end
    metric_for_best_model="accuracy",
    no_cuda=False  # Ensure the GPU is used if available
)

# Load metric
metric = load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model_save_path = 'models/sentiment_model'
os.makedirs(model_save_path, exist_ok=True)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model trained and saved successfully at {model_save_path}!")
