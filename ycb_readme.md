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
  --dataset-path /data1/ycb/datasets/pick_and_place_apple_bidirectional_brainco_60/lerobot_v2.1 \
  --embodiment-tag UNITREE_G1_UPPER_BODY \
  --modality-config-path examples/G1/upper_body/unitree_g1_upper_body_config.py

# 2. Launch the training script on H200 server
chmod +x train.sh
nohup bash train.sh > train.log 2>&1 < /dev/null &
echo $! > train.pid
disown

# 3. Check the logs during finetune
cat train.pid
ps -fp $(cat train.pid)
tail -f train.log
```
---
### Finetune specific parameters
```bash
gr00t/configs/finetune_config.py
```
