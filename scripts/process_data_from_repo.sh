#!/bin/bash

export PYTHONPATH="$PWD"

python "src/data/read.py" --output_dir="../data/train_new_raw" --path="../data/train_new"
python "src/data/read.py" --output_dir="../data/public_test_raw" --path="../data/public_test" --part="test"
