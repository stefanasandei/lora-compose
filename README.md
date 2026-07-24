# Multi-LoRA Composition in Diffusion Models 

work in progress

## Experiment results

Initially, we train single subject adapters to compare individual-concept methods. For the hyperparameters of each methods, please check its coresponding config file under `config/training`. 

| Method          | Identity $\uparrow$ | Prompt $\uparrow$ | Leakage $\downarrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| --------------- | ------------------- | ----------------- | -------------------- | ----------------------- | ------------------- |
| LoRA            | 0.402               | 0.359             | 0.217                | 0.792                   | 0.597               |
| DreamBooth-LoRA | 0.309               | 0.361             | 0.147                | 0.809                   | 0.531               |
| DoRA            | 0.395               | 0.353             | 0.217                | 0.790                   | 0.591               |
| LoHa            |                     |                   |                      |                         |                     |
| LoKr            | 0.316               | 0.357             | 0.154                | 0.799                   | 0.536               |
| PiSSA           | 0.373               | 0.358             | 0.227                | 0.768                   | 0.569               |
| OFTv2           | 0.505               | 0.357             | 0.351                | 0.737                   | 0.615               |
| COFTv2          | 0.329               | 0.355             | 0.198                | 0.787                   | 0.540               |

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
