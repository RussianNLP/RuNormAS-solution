#!/bin/bash

export PYTHONPATH="$PWD"

python "modules/data/read.py" --output_dir="../data/train_new_raw" --path="../data/train_new" --add_answer_sep
python "modules/data/read.py" --output_dir="../data/public_test_raw" --path="../data/public_test" --part="test" --add_answer_sep
