"""
Generate synthetic annotator consensus prediction data.

Domain: LLM Evaluation — predicting consensus quality ratings from
partial, noisy annotator evaluations.

Hidden structure:
  - 8 latent topics (unlabeled) — each item belongs to one
  - 600 annotators with per-topic reliability profiles
  - True quality label (0-5) determined by item features + topic
  - Annotator labels are noisy observations of true quality
  - Consensus = true quality label (with ~8% irreducible noise)
  - Annotator accuracy depends on topic × expertise interaction
  - ~5% adversarial annotators (systematically wrong on certain classes)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    rng = np.random.RandomState(42)

    N_ITEMS = 12000
    N_ANNOTATORS = 600
    N_TOPICS = 8
    N_CLASSES = 6

    # ---- Topic centroids in feature space ----
    topic_centroids = rng.randn(N_TOPICS, 5) * 1.8

    # ---- Generate items ----
    item_topics = rng.randint(0, N_TOPICS, N_ITEMS)
    item_real_feats = np.zeros((N_ITEMS, 5))
    for i in range(N_ITEMS):
        item_real_feats[i] = topic_centroids[item_topics[i]] + rng.randn(5) * 0.7

    # True label: non-linear function of features + topic + noise
    true_labels = np.zeros(N_ITEMS, dtype=int)
    topic_weights = rng.randn(N_TOPICS, 5) * 0.4
    for i in range(N_ITEMS):
        t = item_topics[i]
        f = item_real_feats[i]
        w = topic_weights[t]
        score = (
            np.dot(f, w)
            + np.sin(f[0] * f[1] * 0.5) * 1.5
            + np.abs(f[2] - f[3]) * 0.6
            + f[4] ** 2 * 0.2 * (-1 if t % 2 == 0 else 1)
            + rng.normal(0, 0.9)
        )
        label = int(np.clip(np.round((score + 4) / 8 * 5), 0, 5))
        true_labels[i] = label

    item_noise_feats = rng.randn(N_ITEMS, 3) * 1.2

    items = pd.DataFrame({"item_id": range(N_ITEMS)})
    items["topic"] = item_topics
    items["true_label"] = true_labels
    for j in range(5):
        items[f"if_{j+1}"] = item_real_feats[:, j].round(4)
    for j in range(3):
        items[f"if_{j+6}"] = item_noise_feats[:, j].round(4)

    # ---- Generate annotators ----
    ann_base_acc = rng.uniform(0.35, 0.92, N_ANNOTATORS)

    adversarial_mask = rng.random(N_ANNOTATORS) < 0.05
    ann_base_acc[adversarial_mask] = rng.uniform(0.08, 0.22, adversarial_mask.sum())

    ann_expertise = np.zeros((N_ANNOTATORS, N_TOPICS), dtype=bool)
    for a in range(N_ANNOTATORS):
        n_expert = rng.randint(2, 5)
        topics = rng.choice(N_TOPICS, n_expert, replace=False)
        ann_expertise[a, topics] = True

    ann_confusion_bias = rng.choice([-2, -1, 1, 2], size=N_ANNOTATORS)

    ann_real_feats = np.zeros((N_ANNOTATORS, 4))
    ann_real_feats[:, 0] = ann_base_acc + rng.normal(0, 0.15, N_ANNOTATORS)
    ann_real_feats[:, 1] = ann_expertise.sum(axis=1) + rng.normal(0, 0.4, N_ANNOTATORS)
    for j in range(2, 4):
        ann_real_feats[:, j] = rng.randn(N_ANNOTATORS) * 0.6
    ann_noise_feats = rng.randn(N_ANNOTATORS, 4) * 1.0

    annotators = pd.DataFrame({"annotator_id": range(N_ANNOTATORS)})
    for j in range(4):
        annotators[f"af_{j+1}"] = ann_real_feats[:, j].round(4)
    for j in range(4):
        annotators[f"af_{j+5}"] = ann_noise_feats[:, j].round(4)

    # ---- Generate annotations ----
    annotation_rows = []
    for i in range(N_ITEMS):
        t = item_topics[i]
        tl = true_labels[i]
        n_ann = rng.randint(5, 9)
        selected = rng.choice(N_ANNOTATORS, n_ann, replace=False)

        for a in selected:
            if ann_expertise[a, t]:
                acc = min(ann_base_acc[a] * 1.3, 0.97)
            else:
                acc = ann_base_acc[a] * 0.45

            if rng.random() < acc:
                label = tl
            else:
                bias = ann_confusion_bias[a]
                offset = rng.choice([-2, -1, 1, 2], p=[0.08, 0.32, 0.32, 0.28])
                if rng.random() < 0.4:
                    offset = bias
                label = int(np.clip(tl + offset, 0, 5))

            annotation_rows.append({
                "item_id": i,
                "annotator_id": int(a),
                "label": label,
            })

    annotations = pd.DataFrame(annotation_rows)

    # ---- Consensus = true label with ~8% noise ----
    consensus = items[["item_id", "true_label"]].copy()
    consensus = consensus.rename(columns={"true_label": "consensus_label"})
    noise_mask = rng.random(N_ITEMS) < 0.08
    consensus.loc[noise_mask, "consensus_label"] = rng.randint(
        0, N_CLASSES, noise_mask.sum()
    )

    # ---- Write raw data ----
    out = Path("raw_data")
    out.mkdir(exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    annotators.to_csv(out / "annotators.csv", index=False)
    annotations.to_csv(out / "annotations.csv", index=False)
    consensus.to_csv(out / "consensus.csv", index=False)

    print(f"Items: {len(items)}")
    print(f"Annotators: {len(annotators)}")
    print(f"Annotations: {len(annotations)} ({len(annotations)/N_ITEMS:.1f} per item)")
    print(f"Adversarial annotators: {adversarial_mask.sum()}")
    print(f"Class distribution:\n{consensus['consensus_label'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
