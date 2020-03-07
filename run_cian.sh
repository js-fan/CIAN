#!/bin/bash
GPUS=0,1
GPU_IDS=(${GPUS//,/ })

# Config dataset path
DATASET=/path/of/VOC2012
image_root=$DATASET/JPEGImages
annotation_root=$DATASET/Annotations
groundtruth_root=$DATASET/extra/SegmentationClassAug

# Config model
model=resnet101_largefov
pretrained=./data/pretrained/resnet-101-0000.params
seeds=./data/Seeds/CIAN_SEEDS
snapshot=./snapshot/CIAN/$model


# ========== Training & Testing ==========
# Train
python ./scripts/train_infer_segment.py --image-root $image_root --label-root $seeds --annotation-root $annotation_root --snapshot $snapshot --model $model --pretrained $pretrained --gpus $GPUS

# Test on val set
( IFS=$'\n'; echo "${!GPU_IDS[*]}" ) | xargs -I{} -P${#GPU_IDS[@]} python ./scripts/train_infer_segment.py --image-root $image_root --annotation-root $annotation_root --snapshot $snapshot --model $model --gpus $GPUS --infer --pid {}
python ./scripts/eval_segment.py --groundtruth-root $groundtruth_root --prediction-root $snapshot/pred_crf



# ========== Retraining & Testing ==========
# Generate new seeds
( IFS=$'\n'; echo "${!GPU_IDS[*]}" ) | xargs -I{} -P${#GPU_IDS[@]} python ./scripts/generate_retrain.py --image-root $image_root --annotation-root $annotation_root --snapshot $snapshot --model $model --gpus $GPUS --pid {}

# Train
seeds=$snapshot/train_aug_pred_crf
python ./scripts/train_infer_segment.py --image-root $image_root --label-root $seeds --annotation-root $annotation_root --snapshot $snapshot --model $model --pretrained $pretrained --gpus $GPUS --retrain

# Test on val set
( IFS=$'\n'; echo "${!GPU_IDS[*]}" ) | xargs -I{} -P${#GPU_IDS[@]} python ./scripts/train_infer_segment.py --image-root $image_root --annotation-root $annotation_root --snapshot $snapshot --model $model --gpus $GPUS --infer --pid {} --retrain
python ./scripts/eval_segment.py --groundtruth-root $groundtruth_root --prediction-root $snapshot'_retrain/pred_crf'

