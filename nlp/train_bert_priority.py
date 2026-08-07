import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import os

# 1. Define Evaluation Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_transformer_priority():
    print("🚀 Loading engineered tickets dataset...")
    df = pd.read_csv("data/processed/engineered_tickets.csv").dropna(subset=['text'])
    
    # Use a subset (e.g., 2,000 samples) for faster fine-tuning demonstration
    df_sample = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    
    X = df_sample['text'].tolist()
    y = df_sample['priority_encoded'].tolist()
    num_labels = len(set(y))

    print(f"Dataset loaded. Total samples: {len(X)} across {num_labels} priority classes.")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Load Tokenizer & Model
    MODEL_NAME = "distilbert-base-uncased"
    print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # 4. Tokenize Data
    print("Tokenizing dataset...")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': y_train
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': y_test
    })

    # 5. Set Training Arguments
    output_dir = "models/priority models/bert_priority"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,              # Short epoch count for quick iteration
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"                 # Disable wandb/external loggers
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. Fine-Tune Model
    print("\n🔥 Starting Transformer Fine-Tuning...")
    trainer.train()

    # 8. Evaluate Best Model
    print("\n📊 Evaluating Best Model on Test Set...")
    eval_results = trainer.evaluate()
    print(f"✅ BERT Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"✅ BERT Weighted F1-Score: {eval_results['eval_f1']:.4f}")

    # 9. Save Model & Tokenizer
    print(f"\nSaving fine-tuned BERT model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved successfully!")

if __name__ == "__main__":
    train_transformer_priority()