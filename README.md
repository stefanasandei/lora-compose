# Multi-LoRA Composition in Diffusion Models 

work in progress

## Experiment results

Initially, we train single subject adapters to compare individual-concept methods. Unless specified, we use rank = 32, alpha = 32 and 200 epochs for all.

| Method | DINO $\uparrow$ | CLIP Score $\uparrow$ | LPIPS $\downarrow$ | ArcFace $\uparrow$ | PRES $\downarrow$ |
| ------ | --------------- | --------------------- | ------------------ | ------------------ | ---------------- |
| LoRA   | 0.792           | 0.320                 | 0.435              | 0.374              | 0.211            |

## Usage

To reproduce the results, run:

```
./scripts/train.sh
```

Which will use by default the `./config/train.yaml` configuration.

## License

This work is under the MIT license. See [LICENSE](./LICENSE) for details.
