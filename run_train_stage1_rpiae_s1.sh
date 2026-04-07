CONFIG_PATH=$1
TRAIN_DATA_PATH=./path/to/imagenet/train
export EXPERIMENT_NAME="experiment_name"
export WANDB_MODE="offline" 
export ENTITY="your_wandb_entity" 
export PROJECT="project"

torchrun --standalone --nproc_per_node=8 \
  src/train_stage1_rpiae_s1.py \
  --config $CONFIG_PATH \
  --data-path $TRAIN_DATA_PATH \
  --results-dir results-sh-ceph/stage1 \
  --image-size 256 --precision bf16 \
  --wandb
