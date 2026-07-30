import os

from torch.utils.data import Dataset

from data import build_cached_dataset, list_dataset_pairs
from .standard import batch_loss, collate


class BalancedJointDataset(Dataset):
    """Interleave subjects and give each one equal samples per epoch."""

    def __init__(self, datasets):
        self.datasets = datasets
        self.samples_per_subject = max(map(len, datasets))

    def __getitem__(self, index):
        subject_index = index % len(self.datasets)
        sample_index = index // len(self.datasets)
        dataset = self.datasets[subject_index]
        return dataset[sample_index % len(dataset)]

    def __len__(self):
        return self.samples_per_subject * len(self.datasets)


def prepare_dataset(pipe, cfg, _image_dir):
    subjects = list(cfg.get("characters", []))
    if len(subjects) < 2:
        raise ValueError("Joint training requires at least two `characters`")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Joint training characters must be unique")

    datasets = []
    num_images = 0
    for subject in subjects:
        image_dir = os.path.join(cfg.dataset_dir, subject)
        pairs = list_dataset_pairs(image_dir)
        if not pairs:
            raise ValueError(
                f"No training image/caption pairs found for {subject}: "
                f"{image_dir}"
            )
        datasets.append(
            build_cached_dataset(
                pipe,
                pairs,
                image_dir,
                "joint",
                max_sequence_length=cfg.max_sequence_length,
            )
        )
        num_images += len(pairs)

    return BalancedJointDataset(datasets), num_images
