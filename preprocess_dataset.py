import torch
from datasets import load_from_disk
from transformers import AutoFeatureExtractor, HubertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torchaudio
import ffmpeg
import io
import os

# import transformers
# import sys
# print("Transformers version:", transformers.__version__)
# print("Python executable:", sys.executable)
# from transformers import TrainingArguments
# help(TrainingArguments)


# Load dataset
dataset = load_from_disk("meld_dataset")

# Load HuBERT feature extractor
processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

# Function to load and convert mp4 to waveform
def load_mp4_audio(path, target_sr=16000):
    try:
        path = os.path.normpath(path)
        out, _ = (
            ffmpeg
            .input(path)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        waveform, _ = torchaudio.load(io.BytesIO(out))
        return waveform.squeeze(0)
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return None

# Apply feature extraction
def preprocess(example):   
    waveform = load_mp4_audio(example["path"])
    if waveform is None:
        print(f"🛑 Skipping {example['path']}")
        return {"input_values": None}  # explicitly return None for filtering
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
    return {
        "input_values": inputs.input_values[0],
        "label": example["label"]
    }

if __name__ == "__main__":
    print("⏳ Preprocessing audio...")
    # Preprocess + filter out failed examples
    dataset = dataset.map(preprocess, remove_columns=["path", "__index_level_0__"])

    dataset = dataset.filter(lambda x: x["input_values"] is not None)

    print("✅ Dataset sizes after preprocessing and filtering:")
    print(f"Train: {len(dataset['train'])}")
    print(f"Validation: {len(dataset['validation'])}")
    print(f"Test: {len(dataset['test'])}")

    dataset.set_format(type="torch", columns=["input_values", "label"])

    dataset.save_to_disk("meld_dataset_preprocessed")
    print("💾 Saved preprocessed dataset to 'meld_dataset_preprocessed'")


    

