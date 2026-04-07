CONFIG_PATH=$1
GPUS_PER_NODE=
NNODES=
RANK=
MASTER_ADDR=
TRAIN_DATA_PATH=./path/to/imagenet/train
RESULTS_DIR=./path/to/results/stage1
MASTER_PORT=12348

export EXPERIMENT_NAME="experiment_name"
export WANDB_MODE="offline" 
export ENTITY="your_wandb_entity" 
export PROJECT="project"

torchrun --nnodes=$NNODES \
 --node_rank=$RANK \
 --nproc_per_node=$GPUS_PER_NODE \
 --master_addr=$MASTER_ADDR \
 --master_port=$MASTER_PORT \
  src/train_stage1_rpiae_s23.py \
  --config $CONFIG_PATH \
  --data-path $TRAIN_DATA_PATH \
  --results-dir $RESULTS_DIR \
  --image-size 256 --precision bf16 \
  --wandb
