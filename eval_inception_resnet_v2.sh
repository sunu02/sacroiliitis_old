#!/bin/bash
# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${HOME}/tmp/sacroiliitis-models/inception_resnet_v2/checkpoints/inception_resnet_v2_2016_08_30.ckpt

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=inception_resnet_v2

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${HOME}/tmp/sacroiliitis-models/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=${HOME}/tmp/sacroiliitis-models/data


# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=sacroiliitis \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}