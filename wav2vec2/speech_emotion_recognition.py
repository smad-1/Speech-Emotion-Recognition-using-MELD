import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from datasets import Dataset
from evaluate import load
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import librosa
import soundfile as sf
import warnings

# Suppress librosa audioread deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

# Set PyTorch CUDA memory management to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MELD dataset
meld_path = "/home/bio/SMS/MELD"  # Adjusted to your Ubuntu project directory
train_csv = os.path.join(meld_path, "train", "train_sent_emo.csv")
dev_csv = os.path.join(meld_path, "dev", "dev_sent_emo.csv")
test_csv = os.path.join(meld_path, "test", "test_sent_emo.csv")

# Read CSV files
train_df = pd.read_csv(train_csv)
dev_df = pd.read_csv(dev_csv)
test_df = pd.read_csv(test_csv)

# Map audio files (FLAC files named as diaX_uttY.flac)
def get_audio_path(row, split):
    return os.path.join(meld_path, split, "audio", f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.flac")

train_df['audio_path'] = train_df.apply(lambda x: get_audio_path(x, "train"), axis=1)
dev_df['audio_path'] = dev_df.apply(lambda x: get_audio_path(x, "dev"), axis=1)
test_df['audio_path'] = test_df.apply(lambda x: get_audio_path(x, "test"), axis=1)

# Filter out missing audio files
def filter_valid_files(df, split):
    valid_files = df['audio_path'].apply(os.path.exists)
    missing_files = df['audio_path'][~valid_files]
    if not missing_files.empty:
        print(f"Warning: {len(missing_files)} audio files missing in {split} split: {missing_files.tolist()}")
    return df[valid_files]

train_df = filter_valid_files(train_df, "train")
dev_df = filter_valid_files(dev_df, "dev")
test_df = filter_valid_files(test_df, "test")

# Encode emotion labels
label_encoder = LabelEncoder()
all_emotions = pd.concat([train_df['Emotion'], dev_df['Emotion'], test_df['Emotion']]).unique()
label_encoder.fit(all_emotions)

train_df['label'] = label_encoder.transform(train_df['Emotion'])
dev_df['label'] = label_encoder.transform(dev_df['Emotion'])
test_df['label'] = label_encoder.transform(test_df['Emotion'])

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['audio_path', 'label']])
dev_dataset = Dataset.from_pandas(dev_df[['audio_path', 'label']])
test_dataset = Dataset.from_pandas(test_df[['audio_path', 'label']])

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=len(label_encoder.classes_)
).to(device)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Preprocess audio function
def preprocess_audio(batch):
    try:
        audio, sr = sf.read(batch['audio_path'])
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        batch['input_values'] = inputs.input_values.squeeze().numpy().astype(np.float32)
    except Exception as e:
        print(f"Error processing {batch['audio_path']}: {str(e)}")
        batch['input_values'] = np.zeros(16000, dtype=np.float32)
    return batch

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_audio, remove_columns=['audio_path'])
dev_dataset = dev_dataset.map(preprocess_audio, remove_columns=['audio_path'])
test_dataset = test_dataset.map(preprocess_audio, remove_columns=['audio_path'])

# Data collator for dynamic padding
def data_collator(features):
    input_values = [torch.tensor(f['input_values'], dtype=torch.float32) for f in features if 'input_values' in f]
    labels = [f['label'] for f in features if 'input_values' in f]
    if not input_values:
        return None
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    return {
        'input_values': input_values,
        'labels': torch.tensor(labels, dtype=torch.long)
    }

# Define metrics
accuracy_metric = load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-emotion-meld",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,  # Reduced to avoid OOM
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,  # No need with reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    logging_steps=10,
    fp16=True if torch.cuda.is_available() else False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Evaluate on test set
results = trainer.evaluate(test_dataset)
print(f"Test set accuracy: {results['eval_accuracy']}")

# Save test accuracy
with open("results/test_accuracy.txt", "w") as f:
    f.write(f"Test Accuracy: {results['eval_accuracy']:.4f}\n")

# Save the model
model.save_pretrained("./wav2vec2-emotion-meld-final")
processor.save_pretrained("./wav2vec2-emotion-meld-final")

# Inference example
def predict_emotion(audio_path):
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return label_encoder.inverse_transform([predicted_id])[0]

# Test inference
sample_audio = test_df['audio_path'].iloc[0]
print(f"Predicted emotion for {sample_audio}: {predict_emotion(sample_audio)}")

# Get predictions on test set
pred_logits = trainer.predict(test_dataset).predictions
pred_labels = np.argmax(pred_logits, axis=-1)
true_labels = test_dataset['label']

# Save predictions vs true labels
with open("results/predictions_vs_truth.txt", "w") as f:
    f.write("Index\tTrue_Label\tPredicted_Label\n")
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        true_label = label_encoder.inverse_transform([true])[0]
        pred_label = label_encoder.inverse_transform([pred])[0]
        f.write(f"{i}\t{true_label}\t{pred_label}\n")

from sklearn.metrics import classification_report

# Get human-readable label names
target_names = label_encoder.classes_

# Generate classification report
report = classification_report(true_labels, pred_labels, target_names=target_names)

# Save classification report
with open("results/classification_report.txt", "w") as f:
    f.write(report)
