# train_roberta.py
import os
from datasets import load_dataset, Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# SETTINGS - edit paths & hyperparams
DATA_CSV = "../data/fake_combined.csv"   # combine FakeNewsNet + GossipCop + Politifact
OUTPUT_DIR = "../models/roberta_fake"
MODEL_NAME = "roberta-large"
MAX_LEN = 256
BATCH = 4   # reduce if no big GPU
EPOCHS = 3

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary'),
        'precision': precision_score(labels, preds, average='binary'),
        'recall': recall_score(labels, preds, average='binary')
    }

def main():
    # Load CSV
    import pandas as pd
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=['text','label'])
    # optionally subsample to balance classes
    # dataset
    ds = Dataset.from_pandas(df)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds['train']
    val_ds = ds['test']

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    val_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()