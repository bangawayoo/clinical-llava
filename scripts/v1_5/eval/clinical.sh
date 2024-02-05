#!/bin/bash
CUDA_VISIBLE_DEVICES="1"

CKPT="llava-v1.5-13b"
LORA_PATH="checkpoints/llava-v1.5-13b-task-lora"
BASE_MODEL_PATH="liuhaotian/llava-v1.5-13b"

python -m llava.eval.model_clinical_loader \
        --model-path $LORA_PATH \
        --model-base $BASE_MODEL_PATH \
        --question-file ./data/clinical-nlp-val.json \
        --image-folder ../clinical/mediqa-m3g-startingkit/images_final/images_valid/ \
        --answers-file ./data/eval/prediction.json \
        --num-chunks 1 \
        --chunk-idx 0 \
        --temperature 0 \
        --conv-mode vicuna_v1 

python -m llava.eval.eval_clinical