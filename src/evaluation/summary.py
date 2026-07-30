import numpy as np
import pandas as pd


METRICS = (
    "CLIP_Score", "DINO_Base_Preservation", "LPIPS_Base_Distance",
    "ArcFace_Target", "Base_ArcFace_Target",
)


def clustered_interval(frame, metric, num_bootstrap=5000, seed=0):
    prompt_means = (
        frame.groupby("prompt_index", sort=False)[metric].mean().dropna().to_numpy()
    )
    if not len(prompt_means):
        return np.nan, np.nan, np.nan, 0
    mean = float(prompt_means.mean())
    if len(prompt_means) == 1:
        return mean, np.nan, np.nan, 1
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        prompt_means, (num_bootstrap, len(prompt_means)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high), len(prompt_means)


def prompt_mean(frame, metric):
    return frame.groupby("prompt_index")[metric].mean().dropna().mean()


def harmonic_mean(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-8, 1.0)
    return len(values) / np.reciprocal(values).sum()


def headline_metrics(samples):
    target = samples[samples.group == "target"]
    other = samples[samples.group == "other_identity"]
    non_target = samples[samples.group != "target"]
    identity = prompt_mean(target, "ArcFace_Target")
    leakage = prompt_mean(other, "ArcFace_Target")
    preservation = prompt_mean(non_target, "DINO_Base_Preservation")
    return pd.DataFrame([{
        "Identity": identity,
        "Prompt_Adherence": prompt_mean(target, "CLIP_Score"),
        "Concept_Leakage": leakage,
        "Model_Preservation": preservation,
        "Balanced_Score": harmonic_mean((identity, 1 - leakage, preservation)),
    }])


def composition_headline_metrics(samples, identities):
    single_identities = identities[identities.subject_count == 1]
    composed_identities = identities[identities.subject_count >= 2]
    composed_samples = samples[samples.subject_count >= 2]
    identity = _assignment_mean(single_identities, "similarity")
    disentanglement = _assignment_mean(composed_identities, "correct")
    prompt = prompt_mean(composed_samples, "CLIP_Score")
    return pd.DataFrame([{
        "Identity": identity,
        "Disentanglement": disentanglement,
        "Prompt_Adherence": prompt,
        "Balanced_Score": harmonic_mean(
            (identity, disentanglement, prompt)
        ),
    }])


def _assignment_mean(frame, metric):
    if frame.empty:
        return np.nan
    return (
        frame.groupby(["prompt_id", "seed"], sort=False)[metric]
        .mean()
        .groupby("prompt_id")
        .mean()
        .mean()
    )


def summarize_identities(identities):
    rows = []
    groups = {
        "single": identities[identities.subject_count == 1],
        "composed": identities[identities.subject_count >= 2],
    }
    for group, frame in groups.items():
        if frame.empty:
            continue
        rows.append({
            "group": group,
            "Identity": _assignment_mean(frame, "similarity"),
            "Disentanglement": _assignment_mean(frame, "correct"),
            "Identity_Margin": _assignment_mean(frame, "identity_margin"),
            "Face_Assignment_Rate": _assignment_mean(frame, "matched"),
            "n_prompts": int(frame.prompt_id.nunique()),
            "n_assignments": len(frame),
        })
    return pd.DataFrame(rows)


def summarize_samples(samples, num_bootstrap):
    rows = []
    for group, frame in samples.groupby("group", sort=False):
        for seed, metric in enumerate(METRICS):
            mean, low, high, prompts = clustered_interval(
                frame, metric, num_bootstrap, seed
            )
            rows.append({
                "group": group, "metric": metric, "mean": mean,
                "ci95_low": low, "ci95_high": high, "n_prompts": prompts,
                "n_images": int(frame[metric].notna().sum()),
            })
        for metric, column in (
            ("Face_Detection_Rate", "Face_Detected"),
            ("Base_Face_Detection_Rate", "Base_Face_Detected"),
        ):
            rows.append({
                "group": group, "metric": metric, "mean": float(frame[column].mean()),
                "ci95_low": np.nan, "ci95_high": np.nan,
                "n_prompts": int(frame.prompt_index.nunique()),
                "n_images": len(frame),
            })
    return pd.DataFrame(rows)
