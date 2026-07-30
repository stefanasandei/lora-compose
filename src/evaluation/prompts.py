import hashlib
import json
from pathlib import Path


def normalize_prompts(prompts):
    """Normalize legacy ``character`` entries to the multi-subject schema."""
    result = []
    for index, original in enumerate(prompts):
        item = dict(original)
        subjects = item.get("subjects")
        if subjects is None:
            subjects = [item["character"]] if item.get("character") else []
        subjects = [subjects] if isinstance(subjects, str) else list(subjects)
        item["id"] = str(item.get("id", index))
        item["subjects"] = subjects
        item["type"] = item.get("type", "character" if subjects else "general")
        if not str(item.get("prompt", "")).strip():
            raise ValueError(f"Evaluation prompt {item['id']!r} is empty")
        if len(subjects) != len(set(subjects)):
            raise ValueError(
                f"Evaluation prompt {item['id']!r} repeats a subject"
            )
        result.append(item)

    ids = [item["id"] for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation prompt IDs must be unique")
    return result


def load_prompts(path, max_subjects=None):
    with Path(path).open(encoding="utf-8") as file:
        prompts = normalize_prompts(json.load(file))
    if max_subjects is not None:
        oversized = [
            item["id"] for item in prompts
            if len(item["subjects"]) > max_subjects
        ]
        if oversized:
            raise ValueError(
                f"Prompts exceed max_subjects={max_subjects}: "
                + ", ".join(oversized)
            )
    return prompts


def prompt_group(prompt, subjects):
    if any(subject in subjects for subject in prompt["subjects"]):
        return "target"
    if prompt["subjects"]:
        return "other_identity"
    return "general"


def prompt_fingerprint(prompts, num_seeds, seed=0, model=None):
    payload = {
        "prompts": prompts,
        "num_seeds": int(num_seeds),
        "seed": int(seed),
        "model": model,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()
