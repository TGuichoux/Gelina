#!/usr/bin/env bash

conda create -n gelina python=3.10 
conda activate gelina

pip install -e .
pip install -e .[gpu] --extra-index-url https://download.pytorch.org/whl/cu118
pip install --editable ./external/causal-conv1d --no-build-isolation --config-settings editable_mode=compat
pip install -e external/Matcha-TTS/
pip install -e external/WavTokenizer/
pip install -e packages/*

conda deactivate

conda create -n gelina-preprocess python=3.10
pip install whisperx
pip install datasets


