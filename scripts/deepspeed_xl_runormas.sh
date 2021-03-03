#! /bin/bash

NUM_GPUS_PER_WORKER=10

gpt_options=" \
       --train-data-path /home/jovyan/runormas/data/train_new_raw/files_train.list \
       --max-files-per-process 20000 \
       --logging-dir=/home/jovyan/runormas/models/xl/runs_v3/ \
       --load /home/jovyan/devices/xl/340000/ \
       --save /home/jovyan/runormas/models/xl/v3/ \
       --tokenizer-path /home/jovyan/devices/dgpt_transformers/gpt3_serving/xl_serving/gpt3_xl \
       --cache-prefix p2 \
       --save-interval 5000 \
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
       --deepspeed_config /home/jovyan/devices/xl/dgpt_transformers/scripts/pretrain/deepspeed_config/gpt3_xl_sparse_2048.json \
"

run_cmd="mpirun --np ${NUM_GPUS_PER_WORKER} python /home/jovyan/devices/xl/dgpt_transformers/deepspeed_megatron/pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
