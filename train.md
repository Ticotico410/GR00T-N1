## GR00T-N1.7 Train / Inference Tutorial

### Activate the vitual environment (uv)
```bash
# Activate the GR00T-N1.7 vitual environment
source .venv/bin/activate
```
---
### Training
```bash
# 1. Generate the statistic value before finetuning (stats.json | relative_stats.json)
python gr00t/data/stats.py \
  --dataset-path /sh/datasets/g1/pick_up_multiple_cushions_brainco_200/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_WBC \
  --modality-config-path examples/G1/wbc/unitree_g1_wbc_config.py

# 2. Launch the training script on the training server (tmux)
tmux new -s train
cd /sh/ycb/model/GR00T-N1
bash train.sh

# 3. Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train
List sessions: tmux ls
```
---
### Finetune specific parameters
```bash
gr00t/configs/finetune_config.py
```
