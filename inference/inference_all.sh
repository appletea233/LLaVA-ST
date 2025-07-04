export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export DECORD_EOF_RETRY_MAX=204800
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"

# use full model
MODEL_PATH=appletea2333/LLaVA-ST-Qwen2-7B
LORA_PATH=""

# use lora model
# MODEL_PATH=checkpoints/stage1
# LORA_PATH="checkpoints/stage2 checkpoints/stage3"

save_dir=eval_output
sub_dir=llava_st_qwen2_7b
chunk_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo model_path
echo $MODEL_PATH
echo lora_path
echo $LORA_PATH

# ST-Align
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" stvg $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" svg $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" elc $save_dir $sub_dir $chunk_num

# REC
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcoco $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcoco_testA $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcoco_testB $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcoco+ $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcoco+_testA $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcoco+_testB $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcocog $save_dir $sub_dir $chunk_num
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" refcocog_test $save_dir $sub_dir $chunk_num

# TVG
bash inference/inference_multiprocess.sh "$MODEL_PATH" "$LORA_PATH" charades_sta $save_dir $sub_dir $chunk_num