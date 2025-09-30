import numpy as np
import librosa
import gradio as gr
import torch
from transformers import HubertForSequenceClassification, AutoFeatureExtractor

# -------------------------------
# Load trained model & processor from local folder
# -------------------------------
model_path = "./"  # current folder with model.safetensors, config.json, preprocessor_config.json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoFeatureExtractor.from_pretrained(model_path)
model = HubertForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float32)
model.to(device)
model.eval()

label_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "fear", 5: "disgust", 6: "surprise"}

# -------------------------------
# Audio helpers
# -------------------------------
def chunk_audio(audio, sr=16000, chunk_sec=10):
    chunk_len = chunk_sec * sr
    return [audio[i:i + chunk_len] for i in range(0, len(audio), chunk_len)]

def classify_emotion(audio_path):
    if audio_path is None:
        return "⚠️ No audio received"

    try:
        # Load audio as mono 16kHz
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        return f"⚠️ Could not read audio: {e}"

    if len(audio_data) == 0:
        return "⚠️ Audio is empty"

    chunks = chunk_audio(audio_data, sr=sr, chunk_sec=10)
    results = []

    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        pred_label = label_map.get(pred_id, "unknown")
        results.append(f"Chunk {i+1}: {pred_label} (conf: {probs[pred_id]:.2f})")

    if not results:
        return "⚠️ No valid audio chunks found."

    # Overall most frequent emotion
    final_pred = max(
        set([r.split(":")[1].split("(")[0].strip() for r in results]),
        key=[r.split(":")[1].split("(")[0].strip() for r in results].count
    )
    results.append(f"\n✅ Overall Predicted Emotion: {final_pred}")

    return "\n".join(results)

# -------------------------------
# Gradio interface
# -------------------------------
iface = gr.Interface(
    fn=classify_emotion,
    inputs=gr.Audio(label="Upload a .wav audio file", type="filepath"),
    outputs="text",
    title="🎤 HuBERT Emotion Classifier",
    description="Upload a short .wav audio clip. Longer audio is split into 10-second chunks.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()
