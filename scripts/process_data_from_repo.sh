#!/bin/bash

export PYTHONPATH="$PWD"

python "modules/data/read.py" --output_dir="../data/train_new_raw9" --path="../data/train_new" --window_size=40
python "modules/data/read.py" --output_dir="../data/public_test_v13" --path="../data/public_test" --part="test" --window_size=40
