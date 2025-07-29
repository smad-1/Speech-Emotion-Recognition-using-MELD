import os
from datasets import load_from_disk
from transformers import AutoFeatureExtractor, HubertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score

# Load dataset and processor
dataset = load_from_disk("/home/bio/SMS/clone/Speech-Emotion-Recognition-using-MELD/meld_dataset_preprocessed")
processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

model = HubertForSequenceClassification.from_pretrained("facebook/hubert-base-ls960", num_labels=7)

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# Training arguments
training_args = TrainingArguments(
    output_dir="hubert_results",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    eval_strategy="epoch"
)

data_collator = DataCollatorWithPadding(processor, padding=True)

#Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("🚀 Starting training...")
trainer.train()

# Evaluate
print("📊 Final test set accuracy:")
results = trainer.evaluate(dataset["test"])
print("📊 Final test set accuracy:", results)

os.makedirs("results_hubert", exist_ok=True)
with open("results_hubert/test_accuracy.txt", "w") as f:
    f.write(f"Test Accuracy: {results}\n")

#save model
model.save_pretrained("./hubert-emotion-meld-final")
processor.save_pretrained("./hubert-emotion-meld-final")

# Get predictions on test set
print("📝 Saving predictions to CSV...")
predictions = trainer.predict(dataset["test"])
pred_labels = predictions.predictions.argmax(-1)

# Get true labels and paths
test_df = dataset["test"].to_pandas()
test_df["predicted_label"] = pred_labels

# Optional: Map back to emotion names
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}
test_df["true_emotion"] = test_df["label"].map(label_map)
test_df["predicted_emotion"] = test_df["predicted_label"].map(label_map)

# Save to CSV
test_df.to_csv("hubert_predictions.csv", index=False)
print("✅ Predictions saved to hubert_predictions.csv")