import os, io, numpy as np, pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Audio, load_from_disk
import hydra
from omegaconf import DictConfig
from common.data.data_utils import normalize, concat_segments
import torch
import torchaudio
from transformers import AutoProcessor
import whisperx



def whisper_map(audio, asr, asr_align):
    transcription = asr.transcribe(audio.cpu().numpy(force=True).flatten().astype(np.float32))
    return {"whisper_segments": transcription['segments'], "text":normalize(concat_segments(transcription['segments']))}


@hydra.main(version_base=None, config_path="../../configs/preprocess", config_name="asr_dataset")
def main(cfg: DictConfig):
    device = cfg.device
 
    dsets = load_from_disk(cfg.save_path).with_format('torch')

    asr = whisperx.load_model(cfg.model,device=device, compute_type="int8")
    asr_align, align_metadata = whisperx.load_align_model(language_code="en", device=device)

    dsets_asr = dsets.map(lambda x: whisper_map(x,asr, asr_align), input_columns='audio', desc='whisper', num_proc=None,writer_batch_size=cfg.writer_batch_size)
    dsets_asr.save_to_disk(cfg.save_path_2)


if __name__ == "__main__":
    main()
