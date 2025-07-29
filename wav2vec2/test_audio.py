from sklearn.calibration import LabelEncoder
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-emotion-meld-final")
model = Wav2Vec2ForSequenceClassification.from_pretrained("./wav2vec2-emotion-meld-final").to(device)

def predict_emotion_from_file(path):
    # Load audio
    waveform, sr = torchaudio.load(path)
    
    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Remove channel dimension and convert to numpy
    audio = waveform.squeeze().numpy()
    
    # Tokenize
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return LabelEncoder.inverse_transform([predicted_id])[0]


print(predict_emotion_from_file("test_audio/mytest.wav"))
