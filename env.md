```bash
conda create -n irasim python==3.10


huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    --local-dir pretrained_models/stabilityai/stable-diffusion-xl-base-1.0 \
    --include "vae/*"
```