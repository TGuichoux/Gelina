import torch, torchaudio
from transformers import AutoFeatureExtractor, WavLMModel
from scipy.spatial.distance import cosine

MODEL_ID = "microsoft/wavlm-large"
SR = 16_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = WavLMModel.from_pretrained(MODEL_ID).to(device).eval()

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)

def embed(wav_1d):
    inp = feature_extractor(wav_1d.squeeze(), sampling_rate=SR, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        hid = model(**inp).last_hidden_state            # [1, T, H] on GPU
    return hid.mean(1).squeeze().cpu().numpy()          # → CPU ndarray

def voice_similarity(file_a, file_b):
    e1 = embed(load_audio(file_a))
    e2 = embed(load_audio(file_b))
    return 1 - cosine(e1, e2)
