from dataclasses import dataclass

import torch


@dataclass
class FlowMatchingInputs:
    noisy_latents: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor


def prepare_inputs(transformer, batch, scheduler):
    device = transformer.device
    dtype = transformer.dtype
    latents = batch["latent"].to(device, dtype=dtype)
    batch_size = latents.size(0)

    noise = torch.randn_like(latents)
    u = torch.rand(batch_size, device=device)
    indices = (u * scheduler["num_train_timesteps"]).long()
    timesteps = scheduler["timesteps"][indices].to(dtype=dtype)
    sigmas = scheduler["sigmas"][indices].to(dtype=dtype).view(batch_size, 1, 1, 1)

    return FlowMatchingInputs(
        noisy_latents=(1.0 - sigmas) * latents + sigmas * noise,
        target=noise - latents,
        timesteps=timesteps * transformer.config.timestep_scale,
    )


def predict(transformer, inputs, prompt_embeds, attention_mask):
    pred = transformer(
        hidden_states=inputs.noisy_latents,
        encoder_hidden_states=prompt_embeds.to(transformer.device, dtype=transformer.dtype),
        timestep=inputs.timesteps,
        encoder_attention_mask=attention_mask.to(transformer.device),
        return_dict=False,
    )[0]

    if pred.shape[1] == 2 * inputs.target.shape[1]:
        pred = pred.chunk(2, dim=1)[0]

    return pred
