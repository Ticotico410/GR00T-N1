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
# 1. Generate stats (re-run when action horizon / modality changes)
cd /sh/ycb/model/GR00T
python gr00t/data/stats.py \
  --dataset-path /sh/datasets/g1/smpl/tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_SMPL \
  --modality-config-path examples/G1/smpl/unitree_g1_smpl_config.py

# 2. Launch training (tmux)
tmux new -s train_gr00t_n1d7
source /sh/ycb/venvs/gr00t_n1d7/bin/activate
cd /sh/ycb/model/GR00T
bash train.sh
```

#### SMPL root 表示模式

```bash
# A) 默认 relative rot6d
bash train.sh

# B) 绝对欧拉角：action quat→xyz Euler，独立学习，单独归一化
USE_RELATIVE_EULER=1 bash train.sh
# 等价: launch_finetune.py ... --use-relative-euler

# C) 增量欧拉角：delta = wrap(action_euler - state_euler)，state.robot_root 也转欧拉
USE_RELATIVE_EULER=1 USE_STATE_EULER=1 bash train.sh
# 等价: launch_finetune.py ... --use-relative-euler --use-state-euler

# D) 也可用 modality config 触发 delta：把 frame 的 ActionConfig.rep 改成 RELATIVE
#     （examples/G1/smpl/unitree_g1_smpl_config.py 第一个 action_configs），再：
USE_RELATIVE_EULER=1 bash train.sh
```

```bash
# Detach / reattach / kill
# Detach: Ctrl-b then d
tmux attach -t train_gr00t_n1d7
tmux ls
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

# rot6d checkpoint（默认训练）
python gr00t/eval/open_loop_eval.py \
  --dataset-path /sh/datasets/g1/smpl/tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_SMPL \
  --model-path /sh/ycb/checkpoints/GR00T_N1d7_g1_100k_smpl_rel_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/GR00T_N1d7_g1_100k_smpl_rel_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/checkpoint-40000 \
  --save_plot_path /sh/ycb/checkpoints/GR00T_N1d7_g1_100k_smpl_rel_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/GR00T_N1d7_g1_100k_smpl_rel_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/checkpoint-40000/open_loop_eval/rot6d/traj_0.jpeg \
  --traj-ids 0 \
  --denoising-steps 4 \
  --action-horizon 50 \
  --steps 1500 \
  --video-backend pyav \
  --relative-root-mode rot6d

# 绝对欧拉 / 增量欧拉 checkpoint：processor 已写入 use_relative_euler / use_state_euler，
# 解码走 unapply；对比空间可仍用 rot6d 或 absolute：
#   --relative-root-mode absolute
#   --relative-root-mode rot6d
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
# --use-relative-euler / --use-state-euler
```
