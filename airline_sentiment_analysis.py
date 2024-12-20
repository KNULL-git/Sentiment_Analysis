# -*- coding: utf-8 -*-
!pip install transformers datasets

import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset

from huggingface_hub import login
login(token="Your Token")
# Load and preprocess the dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Airline Twitter Data.csv")

# Select relevant columns
data = data[['text', 'airline_sentiment']]

# Map sentiment to integers
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
data.loc[:, 'label'] = data['airline_sentiment'].map(sentiment_map)

# Split data into training and validation sets
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

# Load pre-trained tokenizer and model (using DistilBERT for efficiency)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenization function for the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch compatibility
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10  # Log every 10 steps (adjust based on dataset size)
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)

# Inference on new data
def predict_sentiment(text):
    # Move model to the same device as inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input text and move inputs to the correct device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)

    # Calculate probabilities and determine sentiment
    probs = torch.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(probs, dim=1).item()
    return {0: "negative", 1: "neutral", 2: "positive"}[sentiment]


# Example prediction
example_text = "@VirginAmerica What @dhepburn said."
print(f"Text: {example_text}\nPredicted Sentiment: {predict_sentiment(example_text)}")

model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/airline_sentiment_model")
tokenizer.save_pretrained("/content/drive/MyDrive/Colab Notebooks/airline_sentiment_model")

test_data = "@VirginAmerica What @dhepburn said good."
print(f"Text: {test_data}\nPredicted Sentiment: {predict_sentiment(test_data)}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "/content/drive/MyDrive/Colab Notebooks/airline_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the predict_sentiment function
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    # Get model outputs
    outputs = model(**inputs)
    # Apply softmax to get probabilities
    probs = torch.softmax(outputs.logits, dim=1)
    # Get the predicted sentiment
    sentiment = torch.argmax(probs, dim=1).item()
    # Map to sentiment label
    return {0: "negative", 1: "neutral", 2: "positive"}[sentiment]

# Test with an example
test_data = "@VirginAmerica airline provided great services."
print(f"Text: {test_data}\nPredicted Sentiment: {predict_sentiment(test_data)}")

from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Redefine the dataset (if not already available)
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Airline Twitter Data.csv")
data = data[['text', 'airline_sentiment']]
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
data['label'] = data['airline_sentiment'].map(sentiment_map)

# Re-split the dataset into training and validation sets
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# Convert the validation set to a Hugging Face Dataset
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

# Redefine the tokenizer and tokenize the validation dataset
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"  # Replace with your model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

val_dataset = val_dataset.map(tokenize_function, batched=True)
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create DataLoader for validation set
from torch.utils.data import DataLoader

val_dataloader = DataLoader(val_dataset, batch_size=16)

# Evaluation function
def evaluate_model(dataset_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataset_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

# Run evaluation
preds, labels = evaluate_model(val_dataloader)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
conf_matrix = confusion_matrix(labels, preds)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Define class labels
class_labels = ["negative", "neutral", "positive"]

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, class_labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Call the function to plot
plot_confusion_matrix(conf_matrix, class_labels)
