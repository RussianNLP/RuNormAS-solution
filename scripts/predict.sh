#! /bin/bash

total_processes=$1
weights_path=$2

for (( i=0; i < total_processes; i++ ))
do
python predict.py --num_proc="$i" --master_port="600$i" --weights_path="$weights_path" &

done
