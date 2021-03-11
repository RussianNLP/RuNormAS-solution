#! /bin/bash

USE_DEEPSPEED=1 python ../predict_simple.py \
  --deepspeed_config ../src/deepspeed_config/gpt3_xl_sparse_2048.json \
  --save_preds_path ../../test_pred/no_beams \
  --do_sample 0 \
  --weights_path ../../models/xl/v7/20000/mp_rank_00_model_states.pt \
  --path=../../data/public_test \
  --output_dir=../../data/public_test_raw \
  --part=test \
  --tokenizer-path sberbank-ai/rugpt3xl \
  --add_start_sep
