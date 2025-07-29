import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from evaluate import load
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.preprocessing import LabelEncoder
import librosa
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

meld_path = "/home/bio/SMS/MELD"
train_csv = os.path.join(meld_path, "train", "train_sent_emo.csv")
dev_csv = os.path.join(meld_path, "dev", "dev_sent_emo.csv")
test_csv = os.path.join(meld_path, "test", "test_sent_emo.csv")

train_df = pd.read_csv(train_csv)
dev_df = pd.read_csv(dev_csv)
test_df = pd.read_csv(test_csv)

def get_audio_path(row, split):
    return os.path.join(meld_path, split, "audio", f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.flac")

for df, split in zip([train_df, dev_df, test_df], ["train", "dev", "test"]):
    df['audio_path'] = df.apply(lambda x: get_audio_path(x, split), axis=1)

def filter_valid_files(df):
    valid = df['audio_path'].apply(lambda x: os.path.exists(x) and os.path.getsize(x) > 0)
    return df[valid]

train_df = filter_valid_files(train_df)
dev_df = filter_valid_files(dev_df)
test_df = filter_valid_files(test_df)

label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([train_df['Emotion'], dev_df['Emotion'], test_df['Emotion']]))

train_df['label'] = label_encoder.transform(train_df['Emotion'])
dev_df['label'] = label_encoder.transform(dev_df['Emotion'])
test_df['label'] = label_encoder.transform(test_df['Emotion'])

train_dataset = Dataset.from_pandas(train_df[['audio_path', 'label']])
dev_dataset = Dataset.from_pandas(dev_df[['audio_path', 'label']])
test_dataset = Dataset.from_pandas(test_df[['audio_path', 'label']])

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self",
    num_labels=len(label_encoder.classes_)
).to(device)
model.gradient_checkpointing_enable()

MAX_AUDIO_LENGTH = 12 * 16000  # 12 seconds max

def preprocess_audio(batch):
    try:
        audio, _ = librosa.load(batch['audio_path'], sr=16000)
        if len(audio) > MAX_AUDIO_LENGTH:
            audio = audio[:MAX_AUDIO_LENGTH]
        batch["input_values"] = audio
        return batch
    except Exception as e:
        print(f"Failed to load {batch['audio_path']}: {e}")
        return None

train_dataset = train_dataset.map(preprocess_audio, remove_columns=["audio_path"])
dev_dataset = dev_dataset.map(preprocess_audio, remove_columns=["audio_path"])
test_dataset = test_dataset.map(preprocess_audio, remove_columns=["audio_path"])

train_dataset = train_dataset.filter(lambda x: x is not None)
dev_dataset = dev_dataset.filter(lambda x: x is not None)
test_dataset = test_dataset.filter(lambda x: x is not None)

data_collator = DataCollatorWithPadding(tokenizer=processor, padding=True)

accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./wav2vec2-emotion-meld",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)

def model_input_mapper(batch):
    processed = processor(
        batch["input_values"],
        sampling_rate=16000,
        padding=True,
        truncation=True,
        max_length=192000,
    )
    processed["labels"] = batch["label"]
    return processed

train_dataset = train_dataset.map(model_input_mapper, batched=True)
dev_dataset = dev_dataset.map(model_input_mapper, batched=True)
test_dataset = test_dataset.map(model_input_mapper, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate(test_dataset)
print(f"Test set accuracy: {results['eval_accuracy']}")

os.makedirs("results", exist_ok=True)
with open("results/test_accuracy.txt", "w") as f:
    f.write(f"Test Accuracy: {results['eval_accuracy']:.4f}\n")

model.save_pretrained("./wav2vec2-emotion-meld-final")
processor.save_pretrained("./wav2vec2-emotion-meld-final")
