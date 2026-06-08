## Terminal 1: Launch the inference server

```bash
conda activate gr00t_n1
cd ~/ycb_ws/GR00T-N1

python scripts/inference_service.py --server \
  --model_path ../checkpoints/gr00t_n1_g1_real.PickandPlace/checkpoint-20000 \
  --data_config unitree_g1_wbc \
  --embodiment_tag new_embodiment \
  --denoising_steps 4 \
  --port 5555 \
  --host 0.0.0.0
```

## Terminal 2: Launch the image Server and Dexterous hand Server (Real Mode)

```bash
ssh unitree@192.168.123.164

bash ./start_teleop.sh
```

## Terminal 3: Deploy

```bash
conda activate wbc_pico
cd ~/ycb_ws/GR00T-N1

# Real robot
python scripts/g1_controller.py --real \
  --robot-ip 192.168.123.164 \
  --robot-port 5555 \
  --policy-host localhost \
  --policy-port 5555 \
  --net enp5s0 \
  --warmup-sec 10 \
  --temporal-agg

# Simulation in MeshCat
python scripts/g1_controller.py --sim \
  --episode ../datasets/g1_real.PickandPlace/mcap/mcap/episode_0002.mcap \
  --policy-host localhost \
  --compare-gt \
  --max-frames 10000 \
  --temporal-agg
```


## Replay the training data as a reference
```bash
conda activate wbc_teleop
cd ~/ycb_ws/wic_pico_record

python replay_all_datas_mcap.py \
    --episode ../datasets/g1_real.PickandPlace/mcap/mcap/episode_0000.mcap \
    --eef inspire_hand \
    --mode action \
    --speed 1.0 \
    --all
```
