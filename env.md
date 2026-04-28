```bash
conda create -n irasim python==3.10
bash scripts/install.sh
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    --local-dir pretrained_models/stabilityai/stable-diffusion-xl-base-1.0 \
    --include "vae/*"
```
# Launch
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --rdzv_endpoint 127.0.0.1:29500 --rdzv_id 107 --rdzv_backend c10d main.py --config configs/train/libero/frame_ada.yaml
```