import os
import torch
from datasets import load_from_disk
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Load dataset and processor
dataset = load_from_disk("/home/bio/SMS/clone/Speech-Emotion-Recognition-using-MELD/meld_dataset_preprocessed")
processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

# Preprocess function
def preprocess(batch):
    # Make sure input_values are float32
    batch["input_values"] = [torch.tensor(x, dtype=torch.float32) for x in batch["input_values"]]
    # Labels must be int64 (long)
    batch["labels"] = torch.tensor(batch["label"], dtype=torch.long)
    return batch

# Apply preprocessing
dataset = dataset.map(preprocess, batched=True)

# Load model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=7
)

# Metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# Training arguments
training_args = TrainingArguments(
    output_dir="wav2vec2_results",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    eval_strategy="epoch",
    gradient_accumulation_steps=2,
    fp16=True,
    max_grad_norm=1.0
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# Train
print("🚀 Starting training...")
trainer.train()

# Evaluate
print("📊 Final test set accuracy:")
results = trainer.evaluate(dataset["test"])
print("📊 Final test set accuracy:", results)

os.makedirs("results_wav2vec2", exist_ok=True)
with open("results_wav2vec2/test_accuracy.txt", "w") as f:
    f.write(f"Test Accuracy: {results}\n")

# Save model
model.save_pretrained("./wav2vec2-emotion-meld-final")
processor.save_pretrained("./wav2vec2-emotion-meld-final")

# Predictions
print("📝 Saving predictions to CSV...")
predictions = trainer.predict(dataset["test"])
pred_labels = predictions.predictions.argmax(-1)

# Map true and predicted labels
test_df = dataset["test"].to_pandas()
test_df["predicted_label"] = pred_labels
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}
test_df["true_emotion"] = test_df["label"].map(label_map)
test_df["predicted_emotion"] = test_df["predicted_label"].map(label_map)

# Save CSV
test_df.to_csv("wav2vec2_predictions.csv", index=False)
print("✅ Predictions saved to wav2vec2_predictions.csv")
