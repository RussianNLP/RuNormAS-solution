#! /bin/bash

NUM_GPUS_PER_WORKER=16

gpt_options=" \
       --max-files-per-process 20000 \
       --path=../../data/public_test \
       --output_dir=../../data/public_test_raw \
       --save_preds_path=../../test_pred/beams16 \
       --part=test \
       --num_beams 5 \
       --logging-dir=../../models/xl/runs_v7/ \
       --load ../../models/xl/v7/ \
       --tokenizer-path sberbank-ai/rugpt3xl \
       --cache-prefix p7 \
       --no-load-optim \
       --finetune \
       --log-interval 100 \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --distributed-backend nccl \
       --fp16 \
       --sparse-mode alternating \
       --deepspeed \
       --deepspeed_config ../src/deepspeed_config/gpt3_xl_sparse_2048.json \
"

run_cmd="USE_DEEPSPEED=1 mpirun --np ${NUM_GPUS_PER_WORKER} python ../predict_runormas.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
