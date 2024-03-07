#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"


CKPT="checkpoint-600"
LORA_PATH="./checkpoints/single-image-lora/checkpoint-4000"
# LORA_PATH="liuhaotian/llava-v1.5-13b"
BASE_MODEL_PATH="liuhaotian/llava-v1.5-13b"
QUESTION_FILE="valid"
ANS_FILE="data/eval/prediction.json"

python -m llava.eval.model_clinical_loader \
        --model-path $LORA_PATH \
        --model-base $BASE_MODEL_PATH \
        --answers-file $ANS_FILE \
        --question-file $QUESTION_FILE \
        --image-folder "./" \
        --conv-mode v1

python -m llava.eval.eval_clinical $ANS_FILE


# python -m llava.eval.model_clinical_loader_copy \
#         --model_name_or_path $BASE_MODEL_PATH \
#         --lora_name $LORA_PATH \
#         --output_dir "./data/eval/prediction.json" \
#         --data_path $QUESTION_FILE \
#         --image_folder "./" \
#         --version v1 \
#         --vision_tower openai/clip-vit-large-patch14-336 \
#         --mm_projector_type mlp2x_gelu \
#         --mm_vision_select_layer -2 \
#         --mm_use_im_start_end False \
#         --mm_use_im_patch_token False \
#         --image_aspect_ratio pad \
#         --group_by_modality_length True \
#         --bf16 True 