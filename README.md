# Multi-LoRA Composition in Diffusion Models 

work in progress

## Experiment results

Initially, we train single subject adapters to compare individual-concept methods. Unless specified, we use rank = 32, alpha = 32 and 2400 training steps for all.

| Method     | DINO $\uparrow$ | CLIP Score $\uparrow$ | LPIPS $\downarrow$ | ArcFace $\uparrow$ | PRES $\downarrow$ |
| ---------- | --------------- | --------------------- | ------------------ | ------------------ | ----------------- |
| LoRA       | 0.792           | 0.320                 | 0.435              | 0.374              | 0.210             |
| Dreambooth | 0.809           | 0.317                 | 0.431              | 0.263              | 0.120             |
| OFTv2      | 0.802           | 0.322                 | 0.436              | 0.331              | 0.207             |

## Usage

To reproduce the results, run:

```
./scripts/train.sh
```

Which uses [`config/training/lora.yaml`](./config/training/lora.yaml) by default.

Training configurations live together under `config/training`. To train a DreamBooth LoRA with class-specific prior preservation:

```
./scripts/train.sh --config-name=training/dreambooth_lora
```

## License

This work is under the MIT license. See [LICENSE](./LICENSE) for details.
