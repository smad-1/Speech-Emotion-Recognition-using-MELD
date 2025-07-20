
import io
import os
import ffmpeg
import torchaudio


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

waveform = load_mp4_audio("meld_data/archive/MELD-RAW/MELD.Raw/train/train_splits/dia0_utt0.mp4")
if waveform is not None:
    print("succesful")