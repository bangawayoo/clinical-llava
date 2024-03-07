#!/bin/bash
CUDA_VISIBLE_DEVICES="2,3"

LORA_PATH="./checkpoints/clinical-nlp/checkpoint-400"
BASE_MODEL_PATH="liuhaotian/llava-v1.5-13b"

QUESTION_FILE="../data/clip-annotated-nlp-valid.json"


python -m llava.eval.model_clinical_loader \
        --model-path $LORA_PATH \
        --model-base $BASE_MODEL_PATH \
        --answers-file "./data/eval/prediction.json" \
        --question-file $QUESTION_FILE \
        --image-folder "./" \
        --conv-mode v1

python -m llava.eval.eval_clinical


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
