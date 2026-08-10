## GR00T-N1.7 Train / Inference Tutorial

### Cosmos path split (important)

| Mode | Script | Cosmos path |
|------|--------|-------------|
| **Training (server)** | `train.sh` → `launch_finetune.py` | `/sh/ycb/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/...` |
| **Open-loop (local)** | `eval_open_loop.sh` → `scripts/local_inference_env.sh` | `/home/karthus_chen/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561` |

Checkpoint `config.json` may still embed the server `/sh/ycb/...` absolute path. That is fine: training keeps using `/sh/ycb`; local inference overrides via `COSMOS_REASON2_PATH` without rewriting the checkpoint.

### Paths — training server (keep heavy I/O off `/root`)

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
#    Re-run this whenever action delta_indices / horizon changes (e.g. 16 -> 40).
#    Current sonic dataset: 68-dim action = motion_token(64) + hands(2+2).
cd /sh/ycb/model/GR00T
python gr00t/data/stats.py \
  --dataset-path /sh/datasets/g1/sonic/walk_to_table_and_place_apple_on_pink_plate_100/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path examples/G1/sonic/unitree_g1_sonic_config.py

# 2. Launch the training script on the training server (tmux)
tmux new -s train_gr00t_n1d7
source /sh/ycb/venvs/gr00t_n1d7/bin/activate
# optional: wandb login --relogin   # interactive; or: wandb login --relogin "$WANDB_API_KEY"
cd /sh/ycb/model/GR00T
bash train.sh
# Or skip W&B: set WANDB_ENABLED=false in train.sh

# 3. Detach / reattach tmux
Detach: Ctrl-b then d
Reattach: tmux attach -t train_gr00t_n1d7
List sessions: tmux ls

# Kill the training process
pkill -KILL -f 'gr00t/experiment/launch_finetune.py' || true
```
---
### Open-loop eval — on training server (uses `/sh/ycb` Cosmos)
```bash
source /sh/ycb/venvs/gr00t_n1d7/bin/activate
cd /sh/ycb/model/GR00T
export PYTHONPATH=/sh/ycb/model/GR00T:${PYTHONPATH:-}
export HF_HOME=/sh/ycb/.cache/huggingface
export HF_HUB_CACHE=/sh/ycb/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1
# optional pin: export COSMOS_REASON2_PATH=$HF_HUB_CACHE/models--nvidia--Cosmos-Reason2-2B/snapshots/<hash>

python gr00t/eval/open_loop_eval.py \
  --dataset-path /sh/datasets/g1/sonic/walk_to_table_and_place_apple_on_pink_plate_100/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_SONIC \
  --model-path /sh/ycb/checkpoints/GR00T_N1d7_g1_sonic_walk_to_table_place_apple_on_pink_plate_100/GR00T_N1d7_g1_sonic_walk_to_table_place_apple_on_pink_plate_100/checkpoint-20000 \
  --save_plot_path /sh/ycb/checkpoints/GR00T_N1d7_g1_sonic_walk_to_table_place_apple_on_pink_plate_100/open_loop_eval/traj_0.jpeg \
  --traj-ids 0 \
  --denoising-steps 4 \
  --action-horizon 40 \
  --steps 400 \
  --video-backend pyav \
  --root-eval-space absolute
```

### Open-loop — on local machine (uses home NVMe Cosmos)
```bash
cd /home/karthus_chen/ycb_ws/GR00T-N1
bash eval_open_loop.sh
# Sources scripts/local_inference_env.sh → COSMOS_REASON2_PATH under ~/.cache/huggingface/...
```
---
### Finetune specific parameters
```bash
gr00t/configs/finetune_config.py
```
