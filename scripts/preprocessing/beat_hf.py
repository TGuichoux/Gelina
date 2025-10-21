import os, numpy as np, pandas as pd, soundfile as sf
from datasets import Dataset, DatasetDict, Features, Value, Audio, Sequence
import hydra
from omegaconf import DictConfig

def gen_rows(df, wav_dir, npz_dir, split):
    for _id in df[df.type==split]["id"].tolist():
        wav, sr = sf.read(os.path.join(wav_dir, f"{_id}.wav"))

        yield {
            "audio": wav.astype(np.float32),
            "beat_motion": dict(np.load(os.path.join(npz_dir, _id+'.npz'), allow_pickle=True)),
            "meta_data": {"file_id": _id, "sr": sr, "duration": len(wav)/sr}
        }

@hydra.main(version_base=None, config_path="../../configs/preprocess", config_name="push_dataset_hf")
def main(cfg: DictConfig):
    df = pd.read_csv(os.path.join(cfg.root_dir, "train_test_split.csv"))
    wav_dir, npz_dir = os.path.join(cfg.root_dir, "wave16k"), os.path.join(cfg.root_dir, "smplxflame_30")
    dsets = DatasetDict({
        s: Dataset.from_generator(lambda split=s: gen_rows(df, wav_dir, npz_dir, split))
        for s in ["train","test","val","additional"]
    })
    (dsets.push_to_hub(cfg.hf_repo) if cfg.push_to_hub else dsets.save_to_disk(cfg.save_path))

if __name__ == "__main__":
    main()
