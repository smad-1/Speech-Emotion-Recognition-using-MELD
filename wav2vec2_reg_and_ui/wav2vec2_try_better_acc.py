# save as wav2vec2_stable_finetune.py and run
import os
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Dataset balancing (upsample)
# -------------------------------
def balance_dataset_upsample(dataset_dict):
    train_dataset = dataset_dict["train"]
    samples = train_dataset.to_list()

    label_to_samples = defaultdict(list)
    for item in samples:
        label_to_samples[int(item["label"])].append(item)

    target_size = max(len(s) for s in label_to_samples.values())
    balanced_samples = []
    for label, group in label_to_samples.items():
        if len(group) < target_size:
            group += random.choices(group, k=target_size - len(group))
        balanced_samples.extend(group)

    random.shuffle(balanced_samples)
    new_train = Dataset.from_list(balanced_samples)

    return DatasetDict({
        "train": new_train,
        "validation": dataset_dict["validation"],
        "test": dataset_dict["test"]
    })

# -------------------------------
# 2️⃣ Load dataset
# -------------------------------
dataset = load_from_disk("meld_dataset_preprocessed")
dataset = balance_dataset_upsample(dataset)
print("Train class distribution after balancing:", Counter(dataset["train"]["label"]))

# -------------------------------
# 3️⃣ Processor & Model
# -------------------------------
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=7,
    hidden_dropout=0.3,
    attention_dropout=0.3
)

# -------------------------------
# 4️⃣ Metrics
# -------------------------------
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

data_collator = DataCollatorWithPadding(processor, padding=True)

# -------------------------------
# Helper: progressive unfreeze callback
# -------------------------------
class UnfreezeCallback(TrainerCallback):
    """
    - After epoch 1: unfreeze last `n_last` encoder layers (so model can adapt slowly)
    - After epoch 3: unfreeze entire encoder for full fine-tuning
    """
    def __init__(self, model, n_last=2):
        self.model = model
        self.n_last = n_last
        self.did_unfreeze_partial = False
        self.did_unfreeze_all = False

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        # epoch counting: after first epoch state.epoch == 1.0
        if epoch >= 1 and (not self.did_unfreeze_partial):
            try:
                encoder_layers = self.model.wav2vec2.encoder.layers
                n_layers = len(encoder_layers)
                start_idx = max(0, n_layers - self.n_last)
                for i in range(start_idx, n_layers):
                    for p in encoder_layers[i].parameters():
                        p.requires_grad = True
                self.did_unfreeze_partial = True
                print(f"[UnfreezeCallback] Unfroze last {self.n_last} encoder layers at epoch {epoch}.")
            except Exception as e:
                print("[UnfreezeCallback] Could not unfreeze partial layers:", e)

        if epoch >= 3 and (not self.did_unfreeze_all):
            try:
                for p in self.model.wav2vec2.parameters():
                    p.requires_grad = True
                self.did_unfreeze_all = True
                print(f"[UnfreezeCallback] Unfroze entire encoder at epoch {epoch}.")
            except Exception as e:
                print("[UnfreezeCallback] Could not unfreeze all layers:", e)

# -------------------------------
# 5️⃣ Training Phase 1 (head only warmup)
# -------------------------------
for param in model.wav2vec2.parameters():
    param.requires_grad = False

training_args_phase1 = TrainingArguments(
    output_dir="wav2vec2_results_phase1",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args_phase1,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("🚀 Phase 1: Training classification head only (2 epochs)...")
trainer.train()

# -------------------------------
# 6️⃣ Training Phase 2 (full fine-tuning with stabilizers)
# -------------------------------
# Before phase2: unfreeze last few layers gradually is handled by callback, but enable grad for classifier
for p in model.classifier.parameters():
    p.requires_grad = True

training_args_phase2 = TrainingArguments(
    output_dir="wav2vec2_results_finetune",
    learning_rate=5e-6,               # much smaller LR for stability
    weight_decay=0.01,                # L2 regularization
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",   # monitor validation loss
    greater_is_better=False,             # for loss lower is better
    logging_dir="./logs",
    report_to="none",
    warmup_steps=500,
    lr_scheduler_type="cosine",     # smooth decay
    gradient_accumulation_steps=2,
    max_grad_norm=1.0
)

# Add EarlyStopping on eval_loss


# UnfreezeCallback: unfreeze last 2 layers after epoch 1, full after epoch 3
unfreeze_cb = UnfreezeCallback(model=model, n_last=2)

trainer = Trainer(
    model=model,
    args=training_args_phase2,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[unfreeze_cb]
)

print("🚀 Phase 2: Fine-tuning full model with stabilizers (10 epochs max)...")
trainer.train()

# -------------------------------
# 7️⃣ Final Evaluation
# -------------------------------
print("📊 Evaluating on test set...")
results = trainer.evaluate(dataset["test"])
print("📊 Final Test Accuracy:", results)

os.makedirs("results2_wav2vec2", exist_ok=True)
with open("results2_wav2vec2/test_accuracy.txt", "w") as f:
    f.write(f"{results}\n")

# Save model and processor
model.save_pretrained("wav2vec2-emotion-meld-final2")
processor.save_pretrained("wav2vec2-emotion-meld-final2")

# Predictions
predictions = trainer.predict(dataset["test"])
pred_labels = predictions.predictions.argmax(-1)

test_df = dataset["test"].to_pandas()
test_df["predicted_label"] = pred_labels
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}
test_df["true_emotion"] = test_df["label"].map(label_map)
test_df["predicted_emotion"] = test_df["predicted_label"].map(label_map)
test_df.to_csv("results2_wav2vec2/wav2vec2_predictions2.csv", index=False)
print("✅ Predictions saved to wav2vec2_predictions2.csv")

# -------------------------------
# 8️⃣ Training Curves (Train vs Val Loss)
# -------------------------------
log_history = pd.DataFrame(trainer.state.log_history)

# Extract epoch logs with loss
loss_logs = log_history[log_history["epoch"].notnull()]

train_epochs = loss_logs[loss_logs["loss"].notnull()]["epoch"].tolist()
train_losses = loss_logs[loss_logs["loss"].notnull()]["loss"].tolist()

eval_epochs = loss_logs[loss_logs["eval_loss"].notnull()]["epoch"].tolist()
eval_losses = loss_logs[loss_logs["eval_loss"].notnull()]["eval_loss"].tolist()

plt.figure(figsize=(8,6))
plt.plot(train_epochs, train_losses, marker='o', label="Train Loss")
plt.plot(eval_epochs, eval_losses, marker='s', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss vs Validation Loss")
plt.legend()
plt.grid()
os.makedirs("results2_wav2vec2", exist_ok=True)
plt.savefig("results2_wav2vec2/train_vs_val_loss_stable.png")
plt.show()

print("📉 Train vs Val loss saved to results2_wav2vec2/train_vs_val_loss_stable.png")
