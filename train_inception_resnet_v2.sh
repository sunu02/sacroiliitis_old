#!/bin/bash
# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${HOME}/tmp/sacroiliitis-models/inception_resnet_v2/checkpoints/inception_resnet_v2_2016_08_30.ckpt

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=inception_resnet_v2

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${HOME}/tmp/sacroiliitis-models/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=${HOME}/tmp/sacroiliitis-models/data


#python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR} \
#  --dataset_name=sacroiliitis \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=${MODEL_NAME} \
#  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
#  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
#  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
#  --max_number_of_steps=1000 \
#  --batch_size=128 \
#  --learning_rate=0.01 \
#  --learning_rate_decay_type=fixed \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=10 \
#  --optimizer=rmsprop \
#  --weight_decay=0.00004
  
  python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=sacroiliitis \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004