import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    CLIPModel,
    Dinov2Model,
)

from .identity import face_embedding
from .prompts import prompt_group


class SampleMetrics:
    def __init__(self, device):
        self.device = device
        self.dino_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )
        self.dino_model = (
            Dinov2Model.from_pretrained("facebook/dinov2-base")
            .to(device)
            .eval()
        )
        self.clip_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        self.clip_model = (
            CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            .to(device)
            .eval()
        )
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device).eval()

    def dino_similarity(self, images):
        inputs = self.dino_processor(
            images, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            embeds = self.dino_model(**inputs).pooler_output
        embeds = F.normalize(embeds, dim=-1)
        return torch.dot(embeds[0], embeds[1]).item()

    def clip_similarity(self, image, prompt):
        inputs = self.clip_processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return F.cosine_similarity(
            outputs.image_embeds, outputs.text_embeds
        ).item()

    def lpips_distance(self, output, reference):
        output_tensor = (
            to_tensor(output.resize((224, 224)))
            .unsqueeze(0)
            .to(self.device)
        )
        reference_tensor = (
            to_tensor(reference.resize((224, 224)))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            return self.lpips(reference_tensor, output_tensor).item()


def evaluate_samples(
    prompts,
    output_dir,
    ref_dir,
    subjects,
    face_app,
    num_seeds,
    image_path,
    device,
    primary_subject_embedding=None,
):
    metrics = SampleMetrics(device)
    rows = []
    for prompt_index, prompt_info in enumerate(prompts):
        prompt = prompt_info["prompt"]
        group = prompt_group(prompt_info, subjects)
        for seed in range(num_seeds):
            output_path = image_path(
                output_dir, prompt_index, seed, num_seeds
            )
            reference_path = image_path(
                ref_dir, prompt_index, seed, num_seeds
            )
            with (
                Image.open(output_path) as output_file,
                Image.open(reference_path) as reference_file,
            ):
                output = output_file.convert("RGB")
                reference = reference_file.convert("RGB")
                output_face = face_embedding(face_app, output)
                reference_face = face_embedding(face_app, reference)
                rows.append({
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_info["id"],
                    "seed": seed,
                    "prompt": prompt,
                    "prompt_type": prompt_info["type"],
                    "subject_count": len(prompt_info["subjects"]),
                    "group": group,
                    "output_path": os.fspath(output_path),
                    "reference_path": os.fspath(reference_path),
                    "CLIP_Score": metrics.clip_similarity(output, prompt),
                    "DINO_Base_Preservation": metrics.dino_similarity(
                        [reference, output]
                    ),
                    "LPIPS_Base_Distance": metrics.lpips_distance(
                        output, reference
                    ),
                    "Face_Detected": output_face is not None,
                    "Base_Face_Detected": reference_face is not None,
                    "ArcFace_Target": _similarity(
                        output_face, primary_subject_embedding
                    ),
                    "Base_ArcFace_Target": _similarity(
                        reference_face, primary_subject_embedding
                    ),
                })
    return pd.DataFrame(rows)


def _similarity(face, reference):
    if face is None or reference is None:
        return np.nan
    return float(np.dot(face, reference))
