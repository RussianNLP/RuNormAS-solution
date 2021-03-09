#! /bin/bash

total_processes=16
weights_path="/home/jovyan/runormas/models/xl/v5/20000/mp_rank_00_model_states.pt"
path="../data/public_test"
output_dir="../data/public_test_raw"
part="test"

run () {
  local idx="$1"
  CUDA_VISIBLE_DEVICES="$idx" python predict.py --num_proc="$idx" \
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
