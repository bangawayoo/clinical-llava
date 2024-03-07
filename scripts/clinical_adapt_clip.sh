#!/bin/bash
wandb online
WANDB_RUNNAME="classification-captions"

DATA_PATH='caption-train'
EVAL_DATA_PATH='caption-valid'
IMAGE_FOLDER="./"
OUTPUT_DIR="./checkpoints/caption"
NUM_EPOCHS=30

TUNE_CLIP=True
TUNE_MM_PROJ=True

SAVE_STEPS=500
SAVE_STRATEGY="steps"
EVAL_STEPS=500
EVAL_STRATEGY="epoch"
#zero2
deepspeed --include localhost:1,2,3 --master_port 29600 llava/train/train_mem.py \
    --lora_enable False --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --tune_vision_tower $TUNE_CLIP \
    --tune_mm_mlp_adapter $TUNE_MM_PROJ 

    # --report_to wandb


# deepspeed --include localhost:1 --master_port 29600 llava/train/train_mem.py \
#     --lora_enable False --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path liuhaotian/llava-v1.5-13b \
#     --version v1 \
#     --data_path $DATA_PATH \
#     --eval_data_path $EVAL_DATA_PATH \
#     --image_folder $IMAGE_FOLDER \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $NUM_EPOCHS \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "steps" \
#     --eval_steps 50 \
#     --save_strategy $SAVE_STRATEGY \
#     --save_steps $SAVE_STEPS \
#     --save_total_limit 3 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --tune_vision_tower $TUNE_CLIP \
#     --tune_mm_mlp_adapter $TUNE_MM_PROJ \
#     --pretrain_mm_mlp_adapter "./checkpoints/caption-tmp2/checkpoint-1/mm_projector.bin" \
#     --pretrain_vision_tower "./checkpoints/caption-tmp2/checkpoint-1/vision_tower.bin"