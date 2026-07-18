## GR00T-N1.7 Train / Inference Tutorial

### Paths (keep heavy I/O off `/root`)

| Role | Path |
|------|------|
| Code / cwd | `/sh/ycb/model/GR00T` |
| Venv | `/sh/ycb/venvs/gr00t_n1d7` |
| HF / torch / uv / tmp / wandb cache | `/sh/ycb/.cache/...` (set by `train.sh`) |
| Base weights | `/sh/ycb/.cache/gr00t_n1d7/GR00T-N1.7-3B` |
| Checkpoints | `/sh/ycb/checkpoints/<EXP_NAME>` |
| Dataset | `/sh/datasets/...` |

`train.sh` redirects `HF_HOME`, `TORCH_HOME`, `TMPDIR`, `UV_CACHE_DIR`, `WANDB_*` away from `/root` and `/tmp` (50G overlay).

### Activate the vitual environment (uv)
```bash
# Activate the GR00T-N1.7 virtual environment
source /sh/ycb/venvs/gr00t_n1d7/bin/activate
```
---
### Training
```bash
# 1. Generate the statistic value before finetuning (stats.json | relative_stats.json)
#    Re-run this whenever action delta_indices / horizon changes (e.g. 16 -> 48).
cd /sh/ycb/model/GR00T
python gr00t/data/stats.py \
  --dataset-path /sh/datasets/g1/pick_up_multiple_cushions_brainco_200/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_WBC \
  --modality-config-path examples/G1/wbc/unitree_g1_wbc_config.py

# 2. Launch the training script on the training server (tmux)
#    If WANDB_ENABLED=true in train.sh, export WANDB_API_KEY in the shell (do NOT commit the key).
export WANDB_API_KEY='你的key'   # https://wandb.ai/authorize
tmux new -s train
cd /sh/ycb/model/GR00T
bash train.sh

# 3. Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train
List sessions: tmux ls

# Kill the training process
pkill -KILL -f 'gr00t/experiment/launch_finetune.py' || true
```
---
### Finetune specific parameters
```bash
gr00t/configs/finetune_config.py
```
