#!/bin/bash
wandb online
WANDB_RUNNAME="mediqa-single-google"

DATA_PATH='google-augment'
EVAL_DATA_PATH='valid'
IMAGE_FOLDER="./"
# output directory should contain lora to ensure projector modules are loaded properly in evaluation
OUTPUT_DIR="./checkpoints/google-augment-filter"
NUM_EPOCHS=3
TUNE_CLIP=False
TUNE_PROJ=False

SAVE_STEPS=100
SAVE_STRATEGY="steps"
EVAL_STEPS=100
EVAL_STRATEGY="steps"

# tuning LM using lora
# freeze CLIP, projector 
deepspeed --include localhost:2,3 --master_port 29600 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --tune_vision_tower $TUNE_CLIP \
    --freeze_mm_mlp_adapter $TUNE_PROJ \
    --pretrain_mm_mlp_adapter "./checkpoints/caption/mm_projector.bin" \
    --pretrain_vision_tower "./checkpoints/caption/vision_tower.bin" \
    --report_to wandb
