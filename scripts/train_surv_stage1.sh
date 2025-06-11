#!/usr/bin/env bash
# Stage-1 survival pre-training (simple proportional hazards)

torchrun --standalone --nproc_per_node=1 src/tabicl/train/run.py \
  --wandb_log True \
  --wandb_project TabICL-Surv \
  --wandb_name SurvStage1 \
  --device cuda \
  --dtype float32 \
  --max_steps 120000 \
  --batch_size 256 \
  --micro_batch_size 4 \
  --lr 3e-4 \
  --scheduler cosine_warmup \
  --warmup_proportion 0.05 \
  --gradient_clipping 1.0 \
  --prior_type survival_scm \
  --prior_device cpu \
  --batch_size_per_gp 4 \
  --min_features 10 \
  --max_features 60 \
  --max_seq_len 1024 \
  --min_train_size 0.2 \
  --max_train_size 0.8 \
  --embed_dim 128 \
  --col_num_blocks 3 \
  --col_nhead 4 \
  --row_num_blocks 3 \
  --row_nhead 8 \
  --row_rope_base 100000 \
  --icl_num_blocks 12 \
  --icl_nhead 4 \
  --ff_factor 2 \
  --norm_first True \
  --checkpoint_dir /my/surv_stage1/ckpt \
  --save_temp_every 50 \
  --save_perm_every 5000 \
  --surv_non_prop_prob 0.0 \
  --surv_competing_prob 0.0
