from datasets import DatasetDict, concatenate_datasets, interleave_datasets, load_dataset
import numpy as np
import pandas as pd
import omegaconf
from collections import defaultdict

'''
    Helper functions to build the datamodule.
    - _tag_sources: tag each dataset with its source id.
    - merge_dataset_dicts: merge multiple DatasetDict objects, based on a merging strategy and mixing probabilities.
    - report_sizes: print a report of the sizes of the original and final datasets.
'''

def _tag_sources(dsets, names, num_proc: int = 4, col: str = "source"):
    """Attach a constant `col` to every split of each DatasetDict."""
    return [
        DatasetDict({
            s: d.map(lambda _: {col: name}, batched=False, num_proc=num_proc, desc='tagging sources')
            for s, d in ds.items()
        })
        for name, ds in zip(names, dsets, strict=True)
    ]


def merge_dataset_dicts(dsets, names, mixing="concat", stopping_strategy="first_exhausted", seed=42, num_workers=4, tag_source=True):
    """
    Parameters
    ----------
    dsets   : list[DatasetDict]
        All DatasetDict objects you want to merge. They must share the same split keys.
    names   : list[str]
        Names of the datasets, used for reporting.
    mixing  : "concat" | "uniform" | dict
        - "concat":   concatenate (old behaviour).
        - "uniform":  interleave with equal probs across input datasets.
        - dict:       per-split float in [0, 1] -> probability to sample *dataset-0*.
                      Remaining probability is spread uniformly across the others.
                      A list/tuple can also be given to specify explicit probabilities
                      for each dataset in that split.
    stopping_strategy: "first_exhausted" | "all_exhausted"
        How to stop interleaving datasets.
        - "first_exhausted": stop when the first dataset is exhausted. (e.g [40h,3000h,200h] -> 40h will be exhausted first)
        - "all_exhausted":   stop when all datasets are exhausted.

    Returns
    -------
    DatasetDict
    """
    if len(dsets) == 1:
        return dsets[0]

    if tag_source:
        print("Detected multiple datasets. Tagging sources ...")
        print(dsets, names)
        dsets = _tag_sources(dsets, names, num_workers)
        print("Done")
    split_keys = dsets[0].keys()
    out = {}

    for split in split_keys:
        if mixing == "concat":
            print("Concatenating datasets for split:", split)
            out[split] = concatenate_datasets([d[split] for d in dsets])

        else:  
            print("Interleaving datasets for split:", split)
            if mixing == "uniform":
                probs = None                    
            else:
                spec = mixing.get(split, None)
                if spec is None:
                    probs = None
                elif isinstance(spec, (list, tuple, omegaconf.listconfig.ListConfig)):
                    if len(spec) != len(dsets):
                        raise ValueError(f"{split}: expected {len(dsets)} probabilities, got {len(spec)}")
                    probs = list(spec)
                else:       
                    print(spec, type(spec))                      
                    p0 = float(spec)
                    rest = (1.0 - p0) / (len(dsets) - 1)
                    probs = [p0] + [rest] * (len(dsets) - 1)

            out[split] = interleave_datasets(
                [d[split] for d in dsets],
                probabilities=probs,
                seed=seed,
                stopping_strategy=stopping_strategy,
            )
    print('Done.')
    return DatasetDict(out)




def report_sizes(src_dsets, final_ds, dur_key="audio_duration", names=None):
    if names is None:
        names = [f"D{i}" for i in range(len(src_dsets))]

    if len(src_dsets) == 1:
        print("\nSplit     |  Hours")
        print("-------------------")
        total = 0.0
        for split, ds in final_ds.items():
            hours = np.asarray(ds[dur_key]).sum() / 3600.0
            print(f"{split:<9}| {hours:>6.2f}")
            total += hours
        print("-------------------")
        print(f"{'TOTAL':<9}| {total:>6.2f}\n")
        return

    rows, final_totals = [], defaultdict(float)

    # original hours
    for idx, d in enumerate(src_dsets):
        label = f"{names[idx]} (original)"
        for split, ds in d.items():
            hours = np.asarray(ds[dur_key]).sum() / 3600.0
            rows.append(dict(src=label, split=split, hours=hours))

    # sampled hours
    for split, ds in final_ds.items():
        by_src = defaultdict(float)
        for src, dur in zip(ds["source"], ds[dur_key]):
            by_src[src] += dur / 3600.0
        for src, hrs in by_src.items():
            label = f"{src} (sampled)"
            rows.append(dict(src=label, split=split, hours=hrs))
            final_totals[split] += hrs

    # final totals
    for split, hrs in final_totals.items():
        rows.append(dict(src="TOTAL", split=split, hours=hrs))

    df = pd.DataFrame(rows).sort_values(["split", "src"])
    print(df.to_markdown(index=False, floatfmt=".2f"))


def load_smpl_dataset(hf_path):
    '''
    Load a dataset dict from Huggingface.
    If multiple subsplits for train, merge them into one. (e.g train.1, train.2, ..., train.n)

    '''
    ds_all = load_dataset(hf_path)
    train_keys = [k for k in ds_all.keys() if k == "train" or k.startswith("train.")]
    other_keys = [k for k in ds_all.keys() if k not in train_keys]
    new_ds = DatasetDict()
    if train_keys:
        new_ds["train"] = concatenate_datasets([ds_all[k] for k in sorted(train_keys, key=lambda x: (x.count("."), x))])
    for k in other_keys:
        new_ds[k] = ds_all[k]
    return new_ds