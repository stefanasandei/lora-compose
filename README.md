# Multi-LoRA Composition in Diffusion Models 

This is a small-scale research project about parameter-efficient finetuning and adapter composition in diffusion models. Specifically, we compare multiple methods for single & multiple adapters, all trained and evaluated on consumer hardware. The goal is to provide useful information for hobbyists finetuning diffusion models at home.

This work can also serve as an experiments bench, we provide a modular system which makes training and experiment tracking straight forward. To test a new method, simply implement the code in a python file in `./src/adapters/` or `./src/recipes/`, add it to the registry and write the config files.

<!-- todo: insert here an image from final trained model -->

Full details can be read in the upcoming [technical report](./todo).

Requirements for all tested methods:
- training must work within 24gb VRAM, and be scalable to larger models (with quantization)
- frozen text encoder with cached text embeddings
- natural language prompts, from VLM generated captions
- little to no additional inference cost

## Experiment results

We used `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` as the backbone for all training and evaluation, due to the limited hardware resources (one RTX 3090). Metrics are relevant only relative to other method's results. We plan to train only the final ensemble of methods, for multi-adapter composition, for Krea 2 or Ideogram V4.

We compare several methods for composition of multiple adapters, while focusing only on approaches that don't add inference overhead. The best result in each column is **bold** and the second-best is <u>underlined</u> within each table:

| Method                | Adapter  | Identity $\uparrow$ | Disentanglement $\uparrow$ | Prompt $\uparrow$ | Balanced $\uparrow$ |
| --------------------- | -------- | ------------------- | -------------------------- | ----------------- | ------------------- |
| Sum                   | LoRA     | 0.157               | 0.078                      | 0.376             | 0.137               |
| Sum                   | DOP-LoRA | 0.167               | 0.057                      | 0.350             | 0.114               |
| Joint training        | OFTv2    | —                   | —                          | —                 | —                   |
| Orthogonal Adaptation | LoRA     | —                   | —                          | —                 | —                   |


<!-- | SSR-Merge             | LoRA    | —                   | —                          | —                 | —                   | -->

For composed prompts, identity is the ArcFace similarity after optimal
subject-to-face assignment, with missing subjects scored as zero.
Disentanglement is the rate at which an expected subject is both above the
identity threshold and the closest configured identity to its assigned face.
Prompt is CLIP alignment on prompts containing two or three subjects. Balanced
is the equal-weight harmonic mean of these three metrics.

Additionally, we train single subject adapters to compare individual-concept training. For the hyperparameters of each methods, please check its coresponding config file under `./config/training`. Each adapter has been trained for at most 2400 steps:

| Method | Identity $\uparrow$ | Prompt $\uparrow$ | Leakage $\downarrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| ------ | ------------------- | ----------------- | -------------------- | ----------------------- | ------------------- |
| LoRA   | 0.402               | <u>0.359</u>      | 0.217                | 0.792                   | 0.597               |
| DoRA   | 0.395               | 0.353             | 0.217                | 0.790                   | 0.591               |
| LoKr   | 0.369               | 0.358             | <u>0.167</u>         | **0.802**               | 0.582               |
| LoHa   | 0.184               | 0.338             | **0.087**            | <u>0.798</u>            | 0.385               |
| PiSSA  | 0.373               | 0.358             | 0.227                | 0.768                   | 0.569               |
| OFTv2  | <u>0.490</u>        | **0.365**         | 0.277                | 0.759                   | **0.633**           |
| COFTv2 | 0.329               | 0.355             | 0.198                | 0.787                   | 0.540               |
| PEANuT | **0.514**           | 0.348             | 0.321                | 0.747                   | <u>0.631</u>        |

Identity and leakage use ArcFace similarity to the trained subject on target and other-identity prompts, respectively. Prompt is CLIP alignment and preservation is paired DINO similarity to the frozen model on non-target prompts. The balanced score is the equal-weight harmonic mean of identity, `1 - leakage`, and preservation. Detailed per-sample results and prompt-bootstrapped confidence intervals are saved by the evaluation script.

Comparison using different training recipes:

| Method           | Identity $\uparrow$ | Prompt $\uparrow$ | Leakage $\downarrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| ---------------- | ------------------- | ----------------- | -------------------- | ----------------------- | ------------------- |
| DreamBooth-LoRA  | 0.309               | **0.361**         | **0.147**            | **0.809**               | 0.531               |
| DOP-LoRA         | <u>0.440</u>        | 0.345             | 0.197                | 0.786                   | <u>0.626</u>        |
| DreamBooth-OFTv2 | 0.415               | <u>0.357</u>      | <u>0.173</u>         | <u>0.791</u>            | 0.614               |
| DOP-OFTv2        | **0.533**           | 0.343             | 0.279                | 0.743                   | **0.650**           |

<details>

<summary>Additional training runs</summary>

| Method                    | Identity $\uparrow$ | Prompt $\uparrow$ | Leakage $\downarrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| ------------------------- | ------------------- | ----------------- | -------------------- | ----------------------- | ------------------- |
| OFTv2 (epochs=100, b=80)  | <u>0.433</u>        | **0.368**         | <u>0.242</u>         | <u>0.782</u>            | <u>0.611</u>        |
| OFTv2 (epochs=200, b=140) | **0.505**           | <u>0.357</u>      | 0.351                | 0.737                   | **0.615**           |
| LoKr (r=128)              | 0.316               | <u>0.357</u>      | **0.154**            | **0.799**               | 0.536               |

</details>

## Usage

To reproduce the results, run:

```bash
./scripts/train.sh
```

Which uses [`config/training/lora.yaml`](./config/training/lora.yaml) by default.

Training configurations live together under `config/training`. To train a DreamBooth LoRA with class-specific prior preservation:

```bash
./scripts/train.sh --config-name=training/dreambooth_lora
```

Run the configured evaluation with:

```bash
./scripts/eval.sh
# or
./scripts/eval.sh --config-name=eval_composed
```

### Implementation details

Adapters are single file python implementation in `./src/adapters`, with a global registry in `./src/adapters/__init__.py`. Once you add a new adapter, you can use create a yaml Hydra config file for your training run. Each adapter follows a common structure (exported functions), read the LoRA implementation for a basic skeleton.

Similarly, we have more abstractions, which can be composed easily from configs:

| Abstraction | Responsibility                                            | Examples                                      |
| ----------- | --------------------------------------------------------- | --------------------------------------------- |
| Recipe      | Construct training examples, batching, and loss           | Standard, DOP, DreamBooth, joint training     |
| Adapter     | Define the trainable parameterization and optimizer hooks | LoRA, OFTv2, Orthogonal LoRA                  |
| Composition | Transform completed adapters into one deployable artifact | Sum, SSR-Merge                                |
| Evaluation  | Sample and measure any completed artifact                 | ArcFace, CLIP, DINO, preservation, efficiency |

## License

This work is under the MIT license. See [LICENSE](./LICENSE) for details.
