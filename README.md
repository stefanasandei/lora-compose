# Multi-LoRA Composition in Diffusion Models 

work in progress

## Experiment results

Initially, we train single subject adapters to compare individual-concept methods. Unless specified, we use rank = 32, alpha = 32 and 2400 training steps for all.

| Method          | Identity $\uparrow$ | Prompt $\uparrow$ | Leakage $\downarrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| --------------- | ------------------- | ----------------- | -------------------- | ----------------------- | ------------------- |
| LoRA            | 0.402               | 0.359             | 0.217                | 0.792                   | 0.597               |
| DreamBooth-LoRA | 0.309               | 0.361             | 0.147                | 0.809                   | 0.531               |
| OFTv2           | 0.505               | 0.357             | 0.351                | 0.737                   | 0.615               |

Identity and leakage use ArcFace similarity to the trained subject on target and other-identity prompts, respectively. Prompt is CLIP alignment and preservation is paired DINO similarity to the frozen model on non-target prompts. The balanced score is the equal-weight harmonic mean of identity, `1 - leakage`, and preservation. Detailed per-sample results and prompt-bootstrapped confidence intervals are saved by the evaluation script.

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
