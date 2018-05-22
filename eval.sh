#!/bin/sh
CHECKPOINT_DIR=${HOME}/tmp/sacroiliitis-models/inception_v3
DATASET_DIR=${HOME}/tmp/sacroiliitis-models/data
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=sacroiliitis \
    --dataset_split_name=validation \
    --model_name=inception_v3
