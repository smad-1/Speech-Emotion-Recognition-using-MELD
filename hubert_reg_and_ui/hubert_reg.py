import os
import random
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoFeatureExtractor,
    HubertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
dataset = load_from_disk("/home/bio/SMS/clone/Speech-Emotion-Recognition-using-MELD/meld_dataset_preprocessed")
print("✅ Dataset loaded. Train size:", len(dataset["train"]))

# -------------------------------
# 2️⃣ Processor & Model
# -------------------------------
processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

from transformers import HubertConfig

config = HubertConfig.from_pretrained(
    "facebook/hubert-base-ls960",
    num_labels=7
)
config.hidden_dropout_prob = 0.3
config.attention_probs_dropout_prob = 0.3

model = HubertForSequenceClassification.from_pretrained(
    "facebook/hubert-base-ls960",
    config=config
)


# Freeze encoder initially
for param in model.hubert.parameters():
    param.requires_grad = False

# -------------------------------
# 3️⃣ Gradual Unfreeze Callback
# -------------------------------
class UnfreezeCallback(TrainerCallback):
    def __init__(self, model, n_last=2):
        self.model = model
        self.n_last = n_last
        self.did_unfreeze_partial = False
        self.did_unfreeze_all = False

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        if epoch >= 1 and not self.did_unfreeze_partial:
            try:
                encoder_layers = self.model.hubert.encoder.layers
                n_layers = len(encoder_layers)
                for i in range(n_layers - self.n_last, n_layers):
                    for param in encoder_layers[i].parameters():
                        param.requires_grad = True
                self.did_unfreeze_partial = True
                print(f"[UnfreezeCallback] Unfroze last {self.n_last} encoder layers at epoch {epoch}.")
            except Exception as e:
                print("[UnfreezeCallback] Failed to unfreeze partial layers:", e)

        if epoch >= 3 and not self.did_unfreeze_all:
            try:
                for param in self.model.hubert.parameters():
                    param.requires_grad = True
                self.did_unfreeze_all = True
                print(f"[UnfreezeCallback] Unfroze all encoder layers at epoch {epoch}.")
            except Exception as e:
                print("[UnfreezeCallback] Failed to unfreeze all layers:", e)

# -------------------------------
# 4️⃣ Metrics
# -------------------------------
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

data_collator = DataCollatorWithPadding(processor, padding=True)

# -------------------------------
# 5️⃣ Training Arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="hubert_results_v2",           # 🔄 updated
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs_v2",                 # 🔄 updated
    warmup_steps=500,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    report_to="none"
)

# -------------------------------
# 6️⃣ Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[UnfreezeCallback(model=model, n_last=2)]
)

# -------------------------------
# 7️⃣ Training
# -------------------------------
print("🚀 Starting HuBERT training with regularization (v2)...")
trainer.train()

# -------------------------------
# 8️⃣ Evaluation
# -------------------------------
print("📊 Evaluating on test set...")
results = trainer.evaluate(dataset["test"])
print("📊 Final test results:", results)

os.makedirs("results_hubert_v2", exist_ok=True)  # 🔄 updated
with open("results_hubert_v2/test_accuracy.txt", "w") as f:
    f.write(f"{results}\n")

# -------------------------------
# 9️⃣ Save Model and Processor
# -------------------------------
model.save_pretrained("hubert-emotion-meld-final-v2")        # 🔄 updated
processor.save_pretrained("hubert-emotion-meld-final-v2")    # 🔄 updated

# -------------------------------
# 🔟 Predictions
# -------------------------------
print("📝 Saving predictions...")
predictions = trainer.predict(dataset["test"])
pred_labels = predictions.predictions.argmax(-1)

test_df = dataset["test"].to_pandas()
test_df["predicted_label"] = pred_labels

label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}
test_df["true_emotion"] = test_df["label"].map(label_map)
test_df["predicted_emotion"] = test_df["predicted_label"].map(label_map)

test_df.to_csv("results_hubert_v2/hubert_predictions_v2.csv", index=False)  # 🔄 updated

print("✅ Predictions saved to results_hubert_v2/hubert_predictions_v2.csv")
