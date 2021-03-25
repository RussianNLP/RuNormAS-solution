#!/bin/bash

export PYTHONPATH="$PWD"

python "modules/data/parse_conllu.py" \
  --output_dir="../data/train_new14/lenta" \
  --path="/home/jovyan/pos/syntax/tagged_texts/"
