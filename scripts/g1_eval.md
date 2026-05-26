Terminal 1: Inference service

```bash
conda activate gr00t_n1

python scripts/inference_service.py --server \
  --model_path checkpoints/gr00t_n1_g1_real.PickandPlace/checkpoint-20000 \
  --data_config unitree_g1_wbc \
  --embodiment_tag new_embodiment \
  --denoising_steps 4 \
  --port 5555 \
  --host 0.0.0.0
```

Terminal 2: 
```bash
conda activate wbc_pico

python scripts/g1_controller.py \
  --net enp5s0 \
  --eef inspire \
  --policy-host localhost \
  --policy-port 5555 \
  --control-hz 30 \
  --action-horizon 16 \
  --language "Pick up the orange bottle and put it in the pink plate."
```