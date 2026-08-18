## NVIDIA GR00T-N1.7

### Training
```bash
# 1. Generate stats (re-run when action horizon / modality changes)
cd /sh/ycb/model/GR00T
python gr00t/data/stats.py \
  --dataset-path /sh/datasets/g1/smpl/tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_SMPL \
  --modality-config-path examples/G1/smpl/unitree_g1_smpl_config.py

# 2. Finetune specific parameters
gr00t/configs/finetune_config.py

# 3. Launch training (tmux)
tmux new -s train_gr00t_n1d7
source /sh/ycb/venvs/gr00t_n1d7/bin/activate
cd /sh/ycb/model/GR00T
bash train.sh

# 4. Check the process
# Detach / reattach / kill
# Detach: Ctrl-b then d
tmux attach -t train_gr00t_n1d7
tmux ls
pkill -KILL -f 'gr00t/experiment/launch_finetune.py' || true
```

### SMPL root 表示模式
```bash
# 1) original smpl (default)
bash train.sh --root-process-mode original

# 2) root to rot6d (82d -> 84d)
bash train.sh --root-process-mode rot6d --action-mode relative # or absolute

# 3) root to delta euler (82d -> 81d)
bash train.sh --root-process-mode delta_euler --action-mode relative # or absolute

# 4) root to euler (82d -> 81d)
bash train.sh --root-process-mode euler --action-mode relative # or absolute
```
---
### Open-loop Evaluation
**Open-loop evaluate your fine-tune:**
```bash
cd ~/ycb_ws/GR00T
source .venv/bin/activate
hf auth login

PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/ycb_ws/GR00T${PYTHONPATH:+:$PYTHONPATH}" \
python gr00t/eval/open_loop_eval.py \
  --dataset-path /home/karthus_chen/ycb_ws/datasets/smpl/tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_SMPL \
  --model-path /home/karthus_chen/ycb_ws/checkpoints/GR00T_N1d7_40k_g1_smpl_rot6d_rel_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/checkpoint-40000 \
  --traj-ids 99 \
  --denoising_steps 4 \
  --action-horizon 50 \
  --steps 1800 \
  --root-process-mode rot6d \
  --rot6d-slerp-alpha 0.8
```
Note: `--root-process-mode` original | trans9d | rot6d | delta_euler | euler

---
### Server-Client Inference (for Deployment)
For real-world deployment or simulation evaluation, use the server-client architecture. The policy runs on a GPU server; a lightweight client sends observations and receives actions over ZMQ.
**Terminal 1 — Launch onboard image server and dexterous hand server:**
```bash
ssh unitree@192.168.123.164
bash ./start_teleop.sh
```

**Terminal 2 — Start the policy server:**
```bash
cd ~/ycb_ws/GR00T
source .venv/bin/activate
hf auth login

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python gr00t/eval/run_gr00t_server.py \
    --model-path /home/karthus_chen/unitree_sh_disk/tools/ycb/checkpoints/GR00T_N1d7_150k_g1_smpl_rot6d_absolute_tidy_the_bed_and_pick_cloth_on_bed_and_put_in_laundry_brainco/checkpoint-20000 \
    --embodiment-tag UNITREE_G1_SMPL \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 6666
```

**Terminal 3 — Run open-loop evaluation as a client:**
```bash
cd ~/ycb_ws/unitree_lerobot
conda activate unitree_lerobot

# 504 fsm with original 36 dim action or trans9d (not implement)
PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/ycb_ws/GR00T${PYTHONPATH:+:$PYTHONPATH}" \
python -m unitree_lerobot.eval_robot.eval_g1_36_brainco_gr00t_n1d7 \
    --policy-host 127.0.0.1 \
    --policy-port 6666 \
    --image-host 192.168.123.164 \
    --image-port 5555 \
    --net enp4s0 \
    --no-headless \
    --visualization \
    --frequency 30 \
    --execute-horizon 48 \
    --ema-alpha 0.25 \
    --task "Pick up all scattered cushions and gather them together olderly." \
    --send-real-robot

# 505 fsm with original SMPL action
PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/ycb_ws/GR00T${PYTHONPATH:+:$PYTHONPATH}" \
python -m unitree_lerobot.eval_robot.eval_g1_smpl_original_brainco_gr00t_n1d7 \
  --policy-host 127.0.0.1 \
  --policy-port 6666 \
  --image-host 192.168.123.164 \
  --image-port 5555 \
  --net enp5s0 \
  --no-headless \
  --visualization \
  --frequency 30 \
  --execute-horizon 50 \
  --task "Tidy up the quilt and then pick up the clothes from the bed and put them into the laundry hamper" \
  --send-real-robot

# 505 fsm with Root2Rot6d
PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/ycb_ws/GR00T${PYTHONPATH:+:$PYTHONPATH}" \
python -m unitree_lerobot.eval_robot.eval_g1_smpl_rot6d_brainco_gr00t_n1d7 \
  --policy-host 127.0.0.1 \
  --policy-port 6666 \
  --image-host 192.168.123.164 \
  --image-port 5555 \
  --net enp5s0 \
  --no-headless \
  --visualization \
  --frequency 30 \
  --execute-horizon 50 \
  --rot6d-slerp-alpha 0.8 \
  --task "Tidy up the quilt and then pick up the clothes from the bed and put them into the laundry hamper" \
  --send-real-robot

# 505 fsm with Root2Euler
PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/ycb_ws/GR00T${PYTHONPATH:+:$PYTHONPATH}" \
python -m unitree_lerobot.eval_robot.eval_g1_smpl_euler_brainco_gr00t_n1d7 \
  --policy-host 127.0.0.1 \
  --policy-port 6666 \
  --image-host 192.168.123.164 \
  --image-port 5555 \
  --net enp5s0 \
  --no-headless \
  --visualization \
  --frequency 30 \
  --execute-horizon 50 \
  --ema-alpha 0.195 \
  --task "Tidy up the quilt and then pick up the clothes from the bed and put them into the laundry hamper" \
  --send-real-robot
```