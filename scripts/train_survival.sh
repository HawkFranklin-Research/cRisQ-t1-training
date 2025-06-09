#!/bin/bash

# This script is for continued pre-training of TabICL on synthetic survival data.

# Set the path to your last training checkpoint
# IMPORTANT: Replace this with the actual path to your checkpoint file.
# This should be the model state before the final classification head was attached,
# often saved as step_{latest}.ckpt from a previous training run.
LAST_CHECKPOINT_PATH="/path/to/your/stage2/checkpoint/dir/step-{latest}.ckpt"

# Set the directory for the new survival model checkpoints
SURVIVAL_CHECKPOINT_DIR="/my/survival/checkpoint/dir"

torchrun --standalone --nproc_per_node=1 /path/to/cRisQ-t1/src/tabicl/train/run.py \
    --wandb_log True \
    --wandb_project cRisQ-t1 \
    --wandb_name Survival-Pretraining \
    --wandb_mode online \
    --device cuda \
    --dtype bfloat16 \
    --max_steps 50000 \
    --batch_size 256 \
    --micro_batch_size 2 \
    --lr 5e-5 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.05 \
    --gradient_clipping 1.0 \
    --prior_type mix_scm \
    --prior_device cpu \
    --batch_size_per_gp 4 \
    --min_features 10 \
    --max_features 100 \
    --max_classes 2 \
    --min_seq_len 1000 \
    --max_seq_len 40000 \
    --log_seq_len True \
    --min_train_size 0.5 \
    --max_train_size 0.9 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --checkpoint_dir $SURVIVAL_CHECKPOINT_DIR \
    --checkpoint_path $LAST_CHECKPOINT_PATH \
    --save_perm_every 1000 \
    --only_load_model True # IMPORTANT: This ensures you only load the model weights, not the optimizer state, which is crucial for changing the task.

echo "Survival pre-training script finished."
