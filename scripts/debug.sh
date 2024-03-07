#!/bin/bash
wandb offline
WANDB_RUNNAME=""

DATA_PATH='caption-train'
EVAL_DATA_PATH='caption-valid'
IMAGE_FOLDER="./"
SAVE_STEPS=1
OUTPUT_DIR="./checkpoints/tmp"
NUM_EPOCHS=3
TUNE_CLIP=True



deepspeed --include localhost:2,3 --master_port 29600 llava/train/train_mem.py \
    --lora_enable False --lora_r 64 --lora_alpha 256 --mm_projector_lr 2e-1 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --learning_rate 2e-1 \
    --weight_decay 0. \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --tune_vision_tower $TUNE_CLIP \
    --freeze_backbone False \
    --tune_mm_mlp_adapter True
