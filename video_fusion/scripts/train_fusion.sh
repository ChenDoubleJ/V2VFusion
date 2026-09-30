# #!/bin/bash

# torchrun \
#   --nnodes=1 \
#   --nproc_per_node=8 \
#   ./video_super_resolution/scripts/train_sr.py \
#   --pretrained_model_path 'pretrained_weight/venhancer_v2.pt' \
#   --train_batch_size 1 \
#   --max_train_steps 15000 \
#   --checkpointing_steps 500 \
#   --learning_rate 5e-5 \
#   --train_data_dir 'dataset/' \
#   --num_frames 30 \
#   --output_dir 'out/'
CUDA_VISIBLE_DEVICES=7 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  ./video_fusion/scripts/train_fusion.py \
  --pretrained_model_path 'pretrained_weight/venhancer_v2.pt' \
  --train_batch_size 1 \
  --max_train_steps 15000 \
  --checkpointing_steps 500 \
  --learning_rate 5e-5 \
  --train_data_dir 'dataset/' \
  --num_frames 30 \
  --output_dir 'out-zh/'
