import torch
import numpy as np
import librosa
import gradio as gr
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# -------------------------------
# Load model and processor
# -------------------------------
model_path = "./"
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Funky emojis for emotions 🎭
label_map = {
    0: "😐 Neutral",
    1: "😄 Happy",
    2: "😡 Angry",
    3: "😭 Sad",
    4: "😱 Fear",
    5: "🤢 Disgust",
    6: "🤯 Surprise"
}

# -------------------------------
# Audio helpers
# -------------------------------
def chunk_audio(audio, sr=16000, chunk_sec=10):
    chunk_len = chunk_sec * sr
    return [audio[i:i + chunk_len] for i in range(0, len(audio), chunk_len)]

def classify_emotion(audio_path):
    if audio_path is None:
        return "⚠️ No audio received 🎧"

    try:
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        return f"⚠️ Could not read audio: {e} 😵"

    if len(audio_data) == 0:
        return "⚠️ Audio is empty 🕳️"

    chunks = chunk_audio(audio_data, sr=sr, chunk_sec=10)
    results = []
    chunk_labels = []

    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = int(np.argmax(logits.cpu().numpy()))
        pred_label = label_map.get(pred_id, "❓ Unknown")
        results.append(f"🎶 Chunk {i+1}: {pred_label}")
        chunk_labels.append(pred_label)

    if not results:
        return "⚠️ No valid audio chunks found 🤔"

    # Majority vote for final prediction
    final_pred = max(set(chunk_labels), key=chunk_labels.count)
    results.append(f"\n🎉 ✅ Overall Predicted Emotion: {final_pred} 🎊")

    return "\n".join(results)

# -------------------------------
# Funky UI with Gradio Blocks
# -------------------------------
with gr.Blocks(css="""
    body {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a1c4fd, #c2e9fb);
        background-size: 300% 300%;
        animation: gradientBG 10s ease infinite;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .gradio-container {
        max-width: 750px; 
        margin: auto; 
        padding: 25px; 
        border-radius: 15px; 
        background: rgba(255,255,255,0.8);
        box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
    }
    h1 {
        color: #ff6f61; 
        text-align: center; 
        font-size: 36px;
    }
    .output-text {
        font-size: 18px; 
        color: #222; 
        padding: 15px;
        border-radius: 12px; 
        background: #fff5e6; 
        border: 2px dashed #ff9f43;
    }
    .gr-button {
        background-color: #ff6f61 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 12px !important;
        font-size: 18px;
        padding: 10px 20px;
    }
    .gr-button:hover {
        background-color: #e55039 !important;
    }
""") as demo:

    gr.Markdown("<h1>🎤 Mood Decoder 🎭</h1>")
    gr.Markdown("✨ Upload or record your voice, and let’s see what emotion your audio carries! ")

    with gr.Row():
        audio_input = gr.Audio(label="🎵 Upload / Record Audio", type="filepath", show_label=True)
    
    predict_btn = gr.Button("🚀 Predict Emotion")
    
    output_text = gr.Textbox(
        label="🔮 Prediction Output",
        interactive=False,
        placeholder="✨ Your emotion results will appear here 🎶",
        elem_classes="output-text"
    )
    
    predict_btn.click(fn=classify_emotion, inputs=audio_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
