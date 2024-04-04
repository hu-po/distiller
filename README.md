# Generative Encoder Evolution via Distillation

Distilled Image Encoder via Generative Evolutionary Code Search using LLM Ensemble

## Distillation Targets

Use pre-trained image encoders to encode a directory of images into sequences of image tokens.

`./embed/run.all.big.sh`

```
clip-vit-large-patch14-336
dinov2-giant
siglip-large-patch16-384
```

`./embed/run.all.small.sh` 

```
clip-vit-base-patch16
dinov2-small
siglip-base-patch16-224
```

## Model Archs

- simple MLP
- simple CNN
- Mamba
  - https://arxiv.org/abs/2312.00752
  - https://arxiv.org/pdf/2403.19888
  - https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f
  - https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
  - https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py
- DiJiang (Discrete Cosine Transform)
  - https://arxiv.org/pdf/2403.19928.pdf
- ViTamin
  - https://github.com/Beckschen/ViTamin/blob/main/ViTamin/models/vitamin.py
  - https://arxiv.org/pdf/2404.02132.pdf

## Running Evolution


```
conda create -n distiller python=3.10
conda activate distiller
pip install -r requirements.txt
python run.py
```

## Citation

```
@misc{distiller-2024,
  title={Generative Encoder Evolution via Distillation}
  author={Hugo Ponte},
  year={2024},
  url={https://github.com/hu-po/distiller}
}
```