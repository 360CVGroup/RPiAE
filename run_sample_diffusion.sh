CONFIG_PATH=$1

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config $CONFIG_PATH \
  --sample-dir samples \
  --precision fp32 \
  --per-proc-batch-size 8 \
  --label-sampling equal
