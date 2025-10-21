import os, io, numpy as np, pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Audio, load_from_disk
import hydra
from omegaconf import DictConfig
from gelina.utils.generation_helpers import wav2text
from common.data.data_utils import normalize
import whisper
import torch
from external.WavTokenizer.encoder.utils import convert_audio
import torchaudio
from external.WavTokenizer.decoder.pretrained import WavTokenizer
from transformers import AutoProcessor
from torchaudio.transforms import Resample

resample = Resample(orig_freq=16000, new_freq=24000)

def wavtokenizer_map(audio, wavtokenizer):
    wav = resample(audio)
    features, discrete_code = wavtokenizer.encode_infer(wav.unsqueeze(0).to('cuda'), bandwidth_id=torch.tensor([0]))
    return {"audio_token": discrete_code}


@hydra.main(version_base=None, config_path="../../configs/preprocess", config_name="tokenize_audio")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = "checkpoints/wavtokenizer/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    model_path = "checkpoints/wavtokenizer/wavtokenizer_medium_speech_320_24k.ckpt"

    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    
    dsets = load_from_disk(cfg.save_path).with_format('torch')
    dsets = dsets.map(lambda x: wavtokenizer_map(x, wavtokenizer), input_columns="audio", desc='tokenization', num_proc=None, writer_batch_size=cfg.writer_batch_size)#, remove_columns="audio")
    dsets.save_to_disk(cfg.save_path_2)


if __name__ == "__main__":
    main()
