#! /bin/bash

NUM_GPUS_PER_WORKER=8

gpt_options=" \
       --train-data-path /home/jovyan/runormas/data/train_new_v6/files.list \
       --max-files-per-process 20000 \
       --logging-dir=/home/jovyan/runormas/models/xl/runs_v7/ \
       --load-huggingface sberbank-ai/rugpt3xl \
       --add_start_sep \
       --save /home/jovyan/runormas/models/xl/v7/ \
       --tokenizer-path sberbank-ai/rugpt3xl \
       --cache-prefix p7 \
       --save-interval 500 \
       --no-load-optim \
       --finetune \
       --log-interval 100 \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --batch-size 2 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 20000 \
       --distributed-backend nccl \
       --lr 0.000015 \
       --warmup 0.0 \
       --lr-decay-style constant \
       --weight-decay 1e-2 \
       --fp16 \
       --sparse-mode alternating \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --deepspeed \
       --deepspeed_config ../src/deepspeed_config/gpt3_xl_sparse_2048.json \
"

run_cmd="USE_DEEPSPEED=1 mpirun --np ${NUM_GPUS_PER_WORKER} python ../pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
