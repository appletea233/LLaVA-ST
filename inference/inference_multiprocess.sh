#!/bin/bash
model_path=$1
lora_path=$2
all_name=$3
save_dir=$4
sub_dir=$5
chunk_num=$6
echo $chunk_num

for (( chunk_id=0; chunk_id<chunk_num; chunk_id++ ))
do
    CUDA_VISIBLE_DEVICES="$chunk_id" python inference/multi_task_inference.py --all_name $all_name --save_dir $save_dir --sub_dir $sub_dir --model_path $model_path --lora_path "$lora_path" --chunk_num $chunk_num --chunk_id $chunk_id &

    # sleep 0.1
done

wait
