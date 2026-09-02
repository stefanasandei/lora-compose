import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment


IDENTITY_COLUMNS = (
    "prompt_index", "prompt_id", "prompt_type", "subject_count", "seed",
    "expected_subject", "face_index", "similarity", "other_similarity",
    "identity_margin", "matched", "correct", "bbox_x1", "bbox_y1",
    "bbox_x2", "bbox_y2",
)


def detected_faces(app, image):
    faces = []
    array = np.asarray(image.convert("RGB"))[:, :, ::-1]
    for index, face in enumerate(app.get(array)):
        embedding = face.embedding / np.linalg.norm(face.embedding)
        faces.append((index, embedding, np.asarray(face.bbox, dtype=float)))
    return faces


def face_embedding(app, image):
    """Return the largest face, preserving the legacy single-face metric."""
    faces = detected_faces(app, image)
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: (
            (face[2][2] - face[2][0]) * (face[2][3] - face[2][1])
        ),
    )[1]


def build_reference_embeddings(app, subject_directories):
    references, counts, ceilings = {}, {}, {}
    for subject, directory in subject_directories.items():
        embeddings = []
        for filename in sorted(os.listdir(directory)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            with Image.open(os.path.join(directory, filename)) as image:
                embedding = face_embedding(app, image)
            if embedding is not None:
                embeddings.append(embedding)
        if not embeddings:
            raise ValueError(f"No faces detected in subject references: {directory}")
        reference = np.mean(embeddings, axis=0)
        reference = reference / np.linalg.norm(reference)
        references[subject] = reference
        counts[subject] = len(embeddings)
        ceilings[subject] = float(np.mean(embeddings @ reference))
    return references, counts, ceilings


def assign_identities(expected, faces, references, threshold):
    """Assign expected subjects to detected faces; missing faces score zero."""
    rows = [{
        "expected_subject": subject, "face_index": pd.NA, "similarity": 0.0,
        "other_similarity": np.nan, "identity_margin": np.nan,
        "matched": False, "correct": False, "bbox_x1": np.nan,
        "bbox_y1": np.nan, "bbox_x2": np.nan, "bbox_y2": np.nan,
    } for subject in expected]
    if not expected or not faces:
        return rows

    scores = np.array([
        [np.dot(references[subject], face[1]) for face in faces]
        for subject in expected
    ])
    reference_names = list(references)
    for subject_index, face_index in zip(
        *linear_sum_assignment(scores, maximize=True)
    ):
        detected_index, embedding, bbox = faces[face_index]
        score = float(scores[subject_index, face_index])
        predicted = max(
            reference_names, key=lambda name: np.dot(references[name], embedding)
        )
        other_scores = [
            np.dot(reference, embedding)
            for name, reference in references.items()
            if name != expected[subject_index]
        ]
        other_similarity = float(max(other_scores)) if other_scores else np.nan
        rows[subject_index].update({
            "face_index": detected_index, "similarity": score, "matched": True,
            "other_similarity": other_similarity,
            "identity_margin": score - other_similarity,
            "correct": score >= threshold and predicted == expected[subject_index],
            "bbox_x1": bbox[0], "bbox_y1": bbox[1],
            "bbox_x2": bbox[2], "bbox_y2": bbox[3],
        })
    return rows


def evaluate_identity_assignments(
    app, prompts, output_dir, num_seeds, references, image_path, threshold
):
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        for seed in range(num_seeds):
            if not prompt["subjects"]:
                continue
            with Image.open(
                image_path(output_dir, prompt_index, seed, num_seeds)
            ) as image:
                faces = detected_faces(app, image)
            for assignment in assign_identities(
                prompt["subjects"], faces, references, threshold
            ):
                rows.append({
                    "prompt_index": prompt_index,
                    "prompt_id": prompt["id"],
                    "prompt_type": prompt["type"],
                    "subject_count": len(prompt["subjects"]),
                    "seed": seed,
                    **assignment,
                })
    return pd.DataFrame(rows, columns=IDENTITY_COLUMNS)
