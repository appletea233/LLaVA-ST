export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=7
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export DECORD_EOF_RETRY_MAX=204801

LLM_VERSION=output/ckpt/llava-qwen2-7b-ov-stage1
VISION_MODEL_VERSION=google/siglip-so400m-patch14-384
CKPT_STAGE_2=output/ckpt/llava-qwen2-7b-ov-stage2
OUTPUT_DIR=output/ckpt/llava-qwen2-7b-ov-stage3

# For multi-node training, please change torchrun params, we use 8 A100-80G x 6 nodes for training
torchrun --nproc_per_node=8 --nnodes=1 llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version qwen_1_5 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --data_path data/stage3_data.yaml \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts "mm_vision_resampler" \
    --mm_resampler_type "fast_slow_resampler" \
    --mm_perceiver_latents 81 \
    --mm_perceiver_latents_fast 9 \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_resampler_lr 2e-5 \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy steps\
    --save_steps 200\
    --save_total_limit 5\
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --stage "3" \
    --load_lora_path ${CKPT_STAGE_2}
