import os
from datasets import load_from_disk
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score
import numpy as np

# -------- user-configurable --------
DATASET_PATH = "/home/bio/SMS/clone/Speech-Emotion-Recognition-using-MELD/meld_dataset_preprocessed"
PRETRAINED = "facebook/wav2vec2-base-960h"   # change to larger/smaller checkpoint if desired
OUTPUT_DIR = "wav2vec2_results"
SAMPLING_RATE = 16000
NUM_LABELS = 7
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
# -----------------------------------

# Load dataset and feature-extractor (processor)
dataset = load_from_disk(DATASET_PATH)
processor = AutoFeatureExtractor.from_pretrained(PRETRAINED)

# Load model for sequence classification
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    PRETRAINED,
    num_labels=NUM_LABELS,
    problem_type="single_label_classification"
)

# (Optional) Freeze the convolutional feature extractor to speed up training / save memory
# Uncomment if you want to only train the classification head initially
# for param in model.wav2vec2.feature_extractor.parameters():
#     param.requires_grad = False

# Prepare label mappings (optional, but useful for saving & for model config)
label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}
model.config.id2label = {str(i): label_map[i] for i in range(NUM_LABELS)}
model.config.label2id = {v: int(k) for k, v in model.config.id2label.items()}

# A robust preprocessing function: will create 'input_values' if not already present.
# It handles datasets that contain an "audio" column with structure {"array": <np.array>, "sampling_rate": ...}
def prepare_batch(batch):
    # if dataset already has 'input_values', just return batch (no-op)
    if "input_values" in batch:
        return batch

    # attempt to extract raw audio arrays from common keys
    # priority: 'audio' -> 'speech' -> 'waveform' -> 'raw_audio' -> assume a list of numpy arrays
    audio_key = None
    for k in ("audio", "speech", "waveform", "raw_audio"):
        if k in batch:
            audio_key = k
            break

    if audio_key is None:
        # nothing to do; return as-is
        return batch

    # Build list of raw arrays
    inputs = []
    for item in batch[audio_key]:
        # if each item is a dict like {"array": np.array, "sampling_rate": int}
        if isinstance(item, dict) and "array" in item:
            arr = item["array"]
            sr = item.get("sampling_rate", SAMPLING_RATE)
            # if sampling rate differs, you may need to resample beforehand; we assume correct SR
            inputs.append(arr)
        else:
            # assume it's already a raw waveform array
            inputs.append(item)

    # Use the processor to get model inputs (no return_tensors here, leave as list)
    proc_outputs = processor(inputs, sampling_rate=SAMPLING_RATE, padding=True, truncation=True)
    # processor returns "input_values" as a list of lists/ndarrays
    batch["input_values"] = [np.asarray(x, dtype=np.float32) for x in proc_outputs["input_values"]]
    return batch

# If 'input_values' not present in train split, map & create them (batched)
if "input_values" not in dataset["train"].column_names:
    # Map over all splits; keep label and other columns
    cols = dataset["train"].column_names
    dataset = dataset.map(prepare_batch, batched=True, remove_columns=None)  # don't remove cols automatically

# Data collator pads dynamically
data_collator = DataCollatorWithPadding(processor, padding=True)

# compute_metrics remains same
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# Training arguments (match your original HuBERT choices; consider enabling fp16 if GPU supports it)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    eval_strategy="epoch",
    #fp16=True,  # uncomment to enable mixed precision if you have an NVIDIA GPU with Ampere or newer
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,            # processor works as tokenizer-equivalent for audio
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("🚀 Starting Wav2Vec2 training...")
trainer.train()

# Evaluate
print("📊 Final test set evaluation:")
results = trainer.evaluate(dataset["test"])
print(results)

os.makedirs("results_wav2vec2", exist_ok=True)
with open("results_wav2vec2/test_accuracy.txt", "w") as f:
    f.write(f"Test results: {results}\n")

# Save model & processor
model.save_pretrained("./wav2vec2-emotion-meld-final")
processor.save_pretrained("./wav2vec2-emotion-meld-final")

# Predictions on test set
print("📝 Saving predictions to CSV...")
predictions = trainer.predict(dataset["test"])
pred_labels = predictions.predictions.argmax(-1)

test_df = dataset["test"].to_pandas()
test_df["predicted_label"] = pred_labels
test_df["true_emotion"] = test_df["label"].map(label_map)
test_df["predicted_emotion"] = test_df["predicted_label"].map(label_map)
test_df.to_csv("wav2vec2_predictions.csv", index=False)
print("✅ Predictions saved to wav2vec2_predictions.csv")
