#! /bin/bash

total_processes=1
weights_path="/home/jovyan/runormas/models/xl/v5/20000/mp_rank_00_model_states.pt"
path="../data/public_test"
output_dir="../data/public_test_raw"
part="test"

run () {
  local idx="$1"
  CUDA_VISIBLE_DEVICES="$idx" USE_DEEPSPEED=1 python predict.py --local_rank="$idx" \
    --master_port="600$idx" \
    --weights_path="$weights_path" \
    --path="$path" \
    --output_dir="$output_dir" \
    --part="$part"
}

for (( i=0; i < total_processes; i++ ))
do
  run $i &
done


wait
