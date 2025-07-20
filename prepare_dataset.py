import os
import pandas as pd
import torchaudio
from datasets import Dataset, DatasetDict
import ffmpeg
import torch
import io

# --------- CONFIG ---------
DATA_PATH = "meld_data/archive/MELD-RAW/MELD.Raw/"
AUDIO_PATH = os.path.join(DATA_PATH, "train", "train_splits")

LABEL_MAP = {
    'neutral': 0,
    'happy': 1,
    'angry': 2,
    'sad': 3,
    'fear': 4,
    'disgust': 5,
    'surprise': 6
}
# --------------------------


# Load and clean CSV
def load_csv(split):
    csv_path = os.path.join(DATA_PATH, f"{split}_sent_emo.csv")
    df = pd.read_csv(csv_path)
    df = df[df['Emotion'].isin(LABEL_MAP.keys())]
    df['path'] = df.apply(
        lambda row: os.path.join(AUDIO_PATH, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"),
        axis=1
    )
    df = df[df['path'].apply(os.path.exists)]
    print(f"✅ Loaded {len(df)} valid rows for split: {split}")
    return df[['path', 'Emotion']]


# Label encoding
def encode_labels(df):
    df['label'] = df['Emotion'].map(LABEL_MAP)
    return df[['path', 'label']]


# Load and convert MP4 to waveform
def load_mp4_audio(path, target_sr=16000):
    try:
        out, _ = (
            ffmpeg
            .input(path)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        waveform, _ = torchaudio.load(io.BytesIO(out))
        return waveform
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return None
    

# Main
if __name__ == "__main__":
    print("📂 Loading CSVs...")
    train_df = encode_labels(load_csv("train/train"))
    val_df = encode_labels(load_csv("dev"))
    test_df = encode_labels(load_csv("test"))

    print("📦 Creating HuggingFace DatasetDict...")
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    print("💾 Saving to disk...")
    dataset.save_to_disk("meld_dataset")

    print("✅ Done! Saved to ./meld_dataset")