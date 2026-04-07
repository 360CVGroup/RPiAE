CONFIG=$1

VAL_DATA_PATH=./path/to/imagenet/val

torchrun --standalone --nproc_per_node=8 \
  src/stage1_sample_ddp_eval.py \
  --config $CONFIG \
  --data-path $VAL_DATA_PATH \
  --sample-dir recon_samples \
  --image-size 256
