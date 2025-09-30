import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score

from datasets import DatasetDict, Dataset
from collections import Counter
import random

# def balance_dataset(dataset_dict):
#     train_dataset = dataset_dict["train"]

#     # Count class frequencies
#     label_counts = Counter([int(label) for label in train_dataset["label"]])
#     max_count = max(label_counts.values())

#     # Group samples by label
#     label_to_samples = {label: [] for label in label_counts}
#     for i in range(len(train_dataset)):
#         label = int(train_dataset[i]["label"])   # ✅ convert tensor -> int
#         label_to_samples[label].append(train_dataset[i])

#     # Upsample
#     balanced_samples = []
#     for label, samples in label_to_samples.items():
#         needed = max_count - len(samples)
#         if needed > 0:
#             samples = samples + random.choices(samples, k=needed)
#         balanced_samples.extend(samples)

#     # Shuffle
#     random.shuffle(balanced_samples)

#     # Create new balanced dataset
#     new_train_dataset = Dataset.from_list(balanced_samples)

#     # Replace only train split
#     return DatasetDict({
#         "train": new_train_dataset,
#         "validation": dataset_dict["validation"],
#         "test": dataset_dict["test"]
#     })
from collections import defaultdict

def balance_dataset(dataset_dict):
    train_dataset = dataset_dict["train"]

    # Convert to list of dicts
    samples = train_dataset.to_list()

    # Group by label
    label_to_samples = defaultdict(list)
    for item in samples:
        label = int(item["label"])  # each item is now a dict
        label_to_samples[label].append(item)

    # Find largest class size
    target_size = max(len(s) for s in label_to_samples.values())

    # Upsample
    balanced_samples = []
    for label, group in label_to_samples.items():
        if len(group) < target_size:
            group = group + random.choices(group, k=target_size - len(group))
        balanced_samples.extend(group)

    random.shuffle(balanced_samples)

    # Make a new balanced train dataset
    new_train = Dataset.from_list(balanced_samples)

    # Return a DatasetDict with balanced train + untouched val/test
    return DatasetDict({
        "train": new_train,
        "validation": dataset_dict["validation"],
        "test": dataset_dict["test"]
    })

# def balance_dataset_downsample(dataset_dict):
#     train_dataset = dataset_dict["train"]

#     # Get class counts
#     label_counts = Counter(train_dataset["label"])
#     min_count = min(label_counts.values())

#     # Group samples by label
#     label_to_samples = {int(label): [] for label in set(train_dataset["label"])}
#     for i in range(len(train_dataset)):
#         label = int(train_dataset[i]["label"])
#         label_to_samples[label].append(train_dataset[i])

#     # Downsample all to the minority count
#     balanced_samples = []
#     for label, samples in label_to_samples.items():
#         if len(samples) > min_count:
#             samples = random.sample(samples, k=min_count)
#         balanced_samples.extend(samples)

#     random.shuffle(balanced_samples)
#     new_train_dataset = Dataset.from_list(balanced_samples)

#     return DatasetDict({
#         "train": new_train_dataset,
#         "validation": dataset_dict["validation"],
#         "test": dataset_dict["test"]
#     })


# Load dataset
dataset = load_from_disk("meld_dataset_preprocessed")
dataset = balance_dataset(dataset)

from collections import Counter
print("Train class distribution after balancing:", Counter(dataset["train"]["label"]))


# Load processor and model
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=7)

# Metric computation
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# TrainingArguments
training_args = TrainingArguments(
    output_dir="wav2vec2_results",
    learning_rate=1e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    warmup_steps=500,
    # save_strategy="epoch",  #disable checkpoint saving
    eval_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./logs",
    report_to="none",  # disable wandb
    # load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Data collator
data_collator = DataCollatorWithPadding(processor, padding=True)

from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append((state.epoch, logs['loss']))
            if 'eval_loss' in logs:
                self.eval_losses.append((state.epoch, logs['eval_loss']))

loss_logger = LossLoggerCallback()

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[loss_logger]
)


# Train
print("🚀 Starting Wav2Vec2 training...")
train_result = trainer.train()

# Evaluate
print("📊 Evaluating on test set...")
results = trainer.evaluate(dataset["test"])
print("📊 Final Test Accuracy:", results)

# Save accuracy to file
os.makedirs("results_wav2vec2", exist_ok=True)
with open("results_wav2vec2/test_accuracy.txt", "w") as f:
    f.write(f"{results}\n")

# Save model
model.save_pretrained("wav2vec2-emotion-meld-final")
processor.save_pretrained("wav2vec2-emotion-meld-final")

# Predictions
predictions = trainer.predict(dataset["test"])
pred_labels = predictions.predictions.argmax(-1)

# Convert test set to DataFrame
test_df = dataset["test"].to_pandas()
test_df["predicted_label"] = pred_labels

label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}
test_df["true_emotion"] = test_df["label"].map(label_map)
test_df["predicted_emotion"] = test_df["predicted_label"].map(label_map)

test_df.to_csv("results_wav2vec2/wav2vec2_predictions.csv", index=False)
print("✅ Predictions saved to wav2vec2_predictions.csv")


# 📈 Plot Training Curves
log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv("results_wav2vec2/logs.csv", index=False)

# After trainer.train()
train_epochs, train_losses = zip(*loss_logger.train_losses)
eval_epochs, eval_losses = zip(*loss_logger.eval_losses)

# Plotting
import matplotlib.pyplot as plt

plt.plot(train_epochs, train_losses, label="Train Loss")
plt.plot(eval_epochs, eval_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (Wav2Vec2)")
plt.legend()
plt.grid()
plt.savefig("results_wav2vec2/loss_curve.png")
plt.show()

# Accuracy Curve
if "eval_accuracy" in log_history.columns:
    plt.figure()
    plt.plot(log_history["epoch"], log_history["eval_accuracy"], label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy (Wav2Vec2)")
    plt.legend()
    plt.grid()
    plt.savefig("results_wav2vec2/accuracy_curve.png")
    plt.show()



# # Plot Loss
# plt.figure()
# plt.plot(log_history["epoch"], log_history["loss"], label="Train Loss")
# if "eval_loss" in log_history.columns:
#     plt.plot(log_history["epoch"], log_history["eval_loss"], label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training & Validation Loss (Wav2Vec2)")
# plt.legend()
# plt.savefig("results_wav2vec2/loss_curve.png")

# # Plot Accuracy
# if "eval_accuracy" in log_history.columns:
#     plt.figure()
#     plt.plot(log_history["epoch"], log_history["eval_accuracy"], label="Validation Accuracy", color="green")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Validation Accuracy (Wav2Vec2)")
#     plt.legend()
#     plt.savefig("results_wav2vec2/accuracy_curve.png")

print("📉 Training curves saved to results_wav2vec2/")
print("Train Losses:", train_losses)
print("Validation Losses:", eval_losses)


import numpy as np
unique, counts = np.unique(pred_labels, return_counts=True)
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}
for u, c in zip(unique, counts):
    print(f"{label_map[u]}: {c}")
