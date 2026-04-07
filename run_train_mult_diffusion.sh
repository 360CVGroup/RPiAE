CONFIG_PATH=$1
GPUS_PER_NODE=
NNODES=
RANK=
MASTER_ADDR=
export EXPERIMENT_NAME="experiment_name"
export WANDB_MODE="offline" 
export ENTITY="your_wandb_entity" 
export PROJECT="project"

TRAIN_DATA_PATH=./path/to/imagenet/train
MASTER_PORT=12348

torchrun --nnodes=$NNODES \
 --node_rank=$RANK \
 --nproc_per_node=$GPUS_PER_NODE \
 --master_addr=$MASTER_ADDR \
 --master_port=$MASTER_PORT \
  src/train_diffusion_rpiae.py \
  --config $CONFIG_PATH \
  --compile \
  --data-path $TRAIN_DATA_PATH \
  --results-dir results-sh-ceph/diffusion \
  --precision bf16 \
  --wandb
