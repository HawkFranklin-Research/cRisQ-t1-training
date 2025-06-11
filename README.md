# cRisQ-t1: Continued Pre-training for Survival Analysis

This repository is a fork of the original [TabICL](https://github.com/soda-inria/tabicl) codebase, specifically adapted for the **continued pre-training** of the TabICL foundation model on a **survival analysis task**. The goal is to fine-tune the model on synthetic data that mimics time-to-event clinical outcomes, making it more suitable for applications in healthcare and transcriptomics, such as the cRisQ project.

---

## 🎯 Core Modifications

The original TabICL model was designed for classification. To adapt it for survival analysis, the following key modifications have been implemented:

1.  **🧬 Synthetic Survival Data Generation**:
    * A new data generator (`src/tabicl/prior/survival.py`) was created.
    * It uses the underlying Structural Causal Model (SCM) from the original paper but transforms its continuous output into a survival format, consisting of:
        * An **event indicator** (binary: 1 for event observed, 0 for censored).
        * An **observed time-to-event** (continuous).
    * This is achieved through a random censoring mechanism applied to the SCM's output.
    * The original classification-focused generators are kept intact under `src/tabicl_prior` for reference and standalone data creation.

2.  **🧠 Multi-Task Prediction Head**:
    * The final `ICLearning` module of the TabICL model has been modified (`src/tabicl/model/learning.py`).
    * It now features two parallel prediction heads that take the final sequence embedding as input:
        * A **classification head** to predict the binary event indicator.
        * A **regression head** to predict the continuous time-to-event.

3.  **📉 Combined Loss Function**:
    * The training loop (`src/tabicl/train/run.py`) now optimizes a composite loss function.
    * The total loss is a weighted sum of:
        * **Binary Cross-Entropy Loss** for the event prediction.
        * **Mean Squared Error (MSE) Loss** for the time prediction.

---

## 🚀 How to Run the Continued Pre-training

This repository is set up for training, not for direct inference. Follow these steps to launch a continued pre-training run.

### 1. Setup the Environment

First, install the necessary dependencies and the `tabicl` package in editable mode.

```bash
# Make sure your virtual environment is activated
pip install -e .
2. Prepare a Base Checkpoint
This code is designed for continued pre-training. You must have a base model checkpoint from which to start. This is typically a checkpoint from Stage 1 or Stage 2 of the original TabICL training curriculum.

3. Configure the Training Script
Open the training script at scripts/train_survival.sh. You must modify the following paths at the top of the file:

Bash

# Set the path to your last training checkpoint
# IMPORTANT: Replace this with the actual path to your checkpoint file.
LAST_CHECKPOINT_PATH="/path/to/your/stage2/checkpoint/dir/step-{latest}.ckpt"

# Set the directory for the new survival model checkpoints
SURVIVAL_CHECKPOINT_DIR="/my/survival/checkpoint/dir"

# Set the path to the main training script
# IMPORTANT: Replace this with the correct path to your repo
TORCHRUN_SCRIPT_PATH="/path/to/your/hawkfranklin-research-crisq-t1-training/src/tabicl/train/run.py"
You can also adjust other hyperparameters in this script, such as learning rate (--lr), batch size (--batch_size), and total steps (--max_steps).
