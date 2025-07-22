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

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MELD dataset
meld_path = "MELD"  # Path to MELD folder
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

# Preprocess audio function
def preprocess_audio(batch):
    try:
        # Load with soundfile to ensure consistent handling of FLAC files
        audio, sr = sf.read(batch['audio_path'])
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Ensure float32 type
        audio = audio.astype(np.float32)
        # Process audio with Wav2Vec2 processor
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        batch['input_values'] = inputs.input_values.squeeze().numpy().astype(np.float32)
    except Exception as e:
        print(f"Error processing {batch['audio_path']}: {str(e)}")
        # Return zero array with float32 type for consistency
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
    if not input_values:  # Handle empty batch
        return None
    # Pad input_values, keep on CPU to allow pin_memory
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    return {
        'input_values': input_values,  # Keep on CPU, Trainer will move to GPU
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
    per_device_train_batch_size=4,  # Reduced to avoid OOM
    per_device_eval_batch_size=4,   # Reduced to avoid OOM
    gradient_accumulation_steps=2,  # Simulate batch size of 8
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