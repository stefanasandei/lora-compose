# Multi-LoRA Composition in Diffusion Models 

This is a small-scale research project about parameter-efficient finetuning and adapter composition in diffusion models. Specifically, we compare multiple methods for single & multiple adapters, all trained and evaluated on consumer hardware. The goal is to provide useful information for hobbyists finetuning diffusion models at home.

This work can also serve as an experiments bench, we provide a modular system which makes training and experiment tracking straightforward. To test a new method, simply implement the code in a python file in `./src/adapters/` or `./src/recipes/`, add it to the registry and create the config files.

<!-- todo: insert here an image from final trained model -->

Full details can be read in the upcoming [technical report](./todo).

Requirements for all tested methods:
- training must work within 24gb VRAM, and be scalable to larger models (with quantization)
- frozen text encoder with cached text embeddings
- natural language prompts, from VLM generated captions
- little to no additional inference cost

## Experiment results

We used `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` as the backbone for all training and evaluation, due to the limited hardware resources (one RTX 3090). Metrics are relevant only relative to other method's results. We plan to train only the final ensemble of methods, for multi-adapter composition, for Krea 2 or Ideogram V4. We ask that these numbers be read as observations from a specific, small-scale setup, not as a ranking: in a noisy eval like this, a small gap doesn't mean one method is always worse.

We compare several methods for composition of multiple adapters, while focusing only on approaches that don't add inference overhead:

| Method    | Adapter  | Identity $\uparrow$ | Disentanglement $\uparrow$ | Prompt $\uparrow$ | Balanced $\uparrow$ |
| --------- | -------- | ------------------- | -------------------------- | ----------------- | ------------------- |
| Sum       | LoRA     | 0.206               | 0.078                      | 0.357             | 0.178               |
| Sum       | DOP-LoRA | 0.256               | 0.057                      | 0.340             | 0.250               |
| SSR Merge | DOP-LoRA | <u>0.325</u>        | <u>0.109</u>               | <u>0.361</u>      | <u>0.275</u>        |
| IterIS    | DOP-LoRA | **0.354**           | **0.167**                  | **0.365**         | **0.336**           |

Identity is normalized against the subject's own reference-photo ceiling, disentanglement is the correct-assignment rate on multi-subject prompts, and balanced is `Identity_norm * sqrt(Preservation)` where preservation is LPIPS similarity to the frozen model on non-target prompts.

<details>

<summary>Additional methods tried</summary>

Methods tried, but resulted in results too poor (due to either constraints or the SANA model): LoRACLR, QR-LoRA, BlockLoRA, Multi-SBoRA. We noticed, in general for this model, that methods which restrict rows of learned matrices result in very poor identity.

Methods whose strength is not fairly represented by this eval:

| Method | Adapter  | Identity $\uparrow$ | Disentanglement $\uparrow$ | Prompt $\uparrow$ | Balanced $\uparrow$ |
| ------ | -------- | ------------------- | -------------------------- | ----------------- | ------------------- |
| TIES   | DOP-LoRA | 0.183               | 0.057                      | 0.352             | 0.179               |

TIES-Merging targets multi-task merging, where fine-tuned task vectors actively conflict in sign; its trim, sign-election, and disjoint merge resolve exactly that interference. This composition eval poses no such sign conflict, so TIES's majority-vote merge only discards per-concept signal that naive sum preserves, and its elementwise operation breaks low rank (requiring SVD re-compression), which this eval fairly penalizes but which the method was never designed to win.

</details>

Additionally, we train single subject adapters to compare individual-concept training. For the hyperparameters of each method, please check its corresponding config file under `./config/training`. Each adapter has been trained for at most 2400 steps:

| Method | Identity $\uparrow$ | Prompt $\uparrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| ------ | ------------------- | ----------------- | ----------------------- | ------------------- |
| LoRA   | 0.491               | 0.359             | 0.717                   | 0.416               |
| NoRA   | 0.523               | <u>0.360</u>      | 0.657                   | 0.423               |
| DoRA   | 0.483               | 0.353             | 0.722                   | 0.411               |
| LoKr   | 0.451               | 0.358             | <u>0.746</u>            | 0.390               |
| LoHa   | 0.225               | 0.339             | **0.779**               | 0.198               |
| PiSSA  | 0.457               | 0.359             | 0.626                   | 0.361               |
| OFTv2  | <u>0.599</u>        | **0.366**         | 0.657                   | <u>0.486</u>        |
| COFTv2 | 0.402               | 0.355             | 0.712                   | 0.339               |
| PEANuT | **0.629**           | 0.349             | 0.604                   | **0.489**           |

Identity is ArcFace similarity to the trained subject on target prompts, normalized against the intrinsic ceiling of the identity metric (the subject's reference photos self-score ~0.819 against their mean embedding). Prompt is CLIP alignment and preservation is LPIPS similarity (`1 - LPIPS_Base_Distance`) to the frozen model on non-target prompts. The balanced score is `Identity_norm * sqrt(Preservation)`. Detailed per-sample results and prompt-bootstrapped confidence intervals are saved by the evaluation script.

Comparison using different training recipes:

| Method           | Identity $\uparrow$ | Prompt $\uparrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| ---------------- | ------------------- | ----------------- | ----------------------- | ------------------- |
| DreamBooth-LoRA  | 0.377               | **0.361**         | <u>0.703</u>            | 0.316               |
| DOP-LoRA         | <u>0.537</u>        | 0.345             | **0.766**               | <u>0.470</u>        |
| DreamBooth-OFTv2 | 0.507               | <u>0.357</u>      | 0.682                   | 0.418               |
| DOP-OFTv2        | **0.651**           | 0.343             | 0.681                   | **0.537**           |

<details>

<summary>Additional training runs</summary>

| Method                    | Identity $\uparrow$ | Prompt $\uparrow$ | Preservation $\uparrow$ | Balanced $\uparrow$ |
| ------------------------- | ------------------- | ----------------- | ----------------------- | ------------------- |
| OFTv2 (epochs=100, b=80)  | 0.529               | **0.368**         | 0.695                   | 0.441               |
| OFTv2 (epochs=200, b=140) | **0.617**           | 0.357             | 0.596                   | **0.477**           |
| LoKr (r=128)              | 0.387               | <u>0.358</u>      | **0.772**               | 0.340               |
| NoRA (r=32, alpha=32)     | 0.431               | 0.358             | 0.694                   | 0.359               |
| LoRA (r=32, alpha=64)     | <u>0.566</u>        | 0.357             | <u>0.702</u>            | <u>0.474</u>        |
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

Adapters are single file python implementations in `./src/adapters`, with a global registry in `./src/adapters/__init__.py`. Once you add a new adapter, you can create a yaml Hydra config file for your training run. Each adapter follows a common structure (exported functions), read the LoRA implementation for a basic skeleton.

Similarly, we have more abstractions, which can be composed easily from configs:

| Abstraction | Responsibility                                            | Examples                                      |
| ----------- | --------------------------------------------------------- | --------------------------------------------- |
| Recipe      | Construct training examples, batching, and loss           | Standard, DOP, DreamBooth, joint training     |
| Adapter     | Define the trainable parameterization and optimizer hooks | LoRA, OFTv2, Orthogonal LoRA                  |
| Composition | Transform completed adapters into one deployable artifact | Sum, SSR-Merge, IterIS                        |
| Evaluation  | Sample and measure any completed artifact                 | ArcFace, CLIP, DINO, preservation, efficiency |

## License

This work is under the MIT license. See [LICENSE](./LICENSE) for details.
