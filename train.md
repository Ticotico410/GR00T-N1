## GR00T-N1.7 Train / Inference Tutorial

### Paths (keep heavy I/O off `/root`)

| Role | Path |
|------|------|
| Code / cwd | `/sh/ycb/model/GR00T` |
| Venv | `/sh/ycb/venvs/gr00t_n1d7` |
| HF / torch / uv / tmp / wandb cache | `/sh/ycb/.cache/...` (set by `train.sh`) |
| Base weights | `/sh/ycb/.cache/gr00t_n1d7/GR00T-N1.7-3B` |
| Cosmos backbone (HF cache) | `/sh/ycb/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/` |
| Checkpoints | `/sh/ycb/checkpoints/<EXP_NAME>` |
| Dataset | `/sh/datasets/...` |

`train.sh` redirects `HF_HOME`, `TORCH_HOME`, `TMPDIR`, `UV_CACHE_DIR`, `WANDB_*` away from `/root` and `/tmp` (50G overlay). Defaults `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so training uses the local HF cache (no Hub pull). Override with `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0` only if you need online download.

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
tmux new -s train
source /sh/ycb/venvs/gr00t_n1d7/bin/activate
# optional: wandb login --relogin   # interactive; or: wandb login --relogin "$WANDB_API_KEY"
cd /sh/ycb/model/GR00T
bash train.sh
# Or skip W&B: set WANDB_ENABLED=false in train.sh

# 3. Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train
List sessions: tmux ls

# Kill the training process
pkill -KILL -f 'gr00t/experiment/launch_finetune.py' || true
```
---
### Open-loop eval (offline plot GT vs pred)
```bash
source /sh/ycb/venvs/gr00t_n1d7/bin/activate
cd /sh/ycb/model/GR00T
export PYTHONPATH=/sh/ycb/model/GR00T:${PYTHONPATH:-}
export HF_HOME=/sh/ycb/.cache/huggingface
export HF_HUB_CACHE=/sh/ycb/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1

# Dataset videos are AV1 → default --video-backend pyav
# Default --root-eval-space absolute keeps original open-loop (full action, absolute space).
# For Unitree root 9D training target space:
#   add --root-eval-space relative9d  (or: ROOT_EVAL_SPACE=relative9d bash eval_open_loop.sh)
python gr00t/eval/open_loop_eval.py \
  --dataset-path /sh/datasets/g1/pick_up_multiple_cushions_brainco_200/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_WBC \
  --model-path /sh/ycb/checkpoints/GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200/GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200/checkpoint-10000 \
  --save_plot_path /sh/ycb/checkpoints/GR00T_N1d7_60k_g1_wbc_pick_up_multiple_cushions_brainco_200/open_loop_eval/traj_100.jpeg \
  --traj-ids 100 \
  --denoising-steps 4 \
  --action-horizon 48 \
  --steps 400 \
  --video-backend pyav \
  --root-eval-space absolute
```
---
### Finetune specific parameters
```bash
gr00t/configs/finetune_config.py
```
