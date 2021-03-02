#!/bin/bash

python "src/read.py" --output_dir="../data/train_raw" --path="../data/train_new"
python "src/read.py" --output_dir="../data/public_test" --path="../data/public_test"
