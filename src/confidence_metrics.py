"""
confidence_metrics.py

Measures how reliable a model's confidence scores are (RQ1).
Works with any model (Keras, TFLite, QAT, PTQ): you pass in the predicted
probabilities (softmax outputs) and the true labels.

Usage:
    from confidence_metrics import confidence_report, attack_report, print_table

    clean = confidence_report(clean_probs, y_test)
    adv = confidence_report(adv_probs, y_test)
    asr = attack_report(clean_probs, adv_probs, y_test)

Note for INT8 TFLite models: convert the raw integer outputs back to
probabilities first, using the output tensor's scale and zero point:
    probs = (raw.astype("float32") - zero_point) * scale
"""

import numpy as np


def _to_class_labels(labels):
    """Accepts integer labels (N,) / (N, 1) or one-hot labels (N, C)."""
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] > 1:
        return labels.argmax(axis=1)
    return labels.reshape(-1).astype(int)


def confidence_report(probs, labels, n_bins=15, high_conf=0.9):
    """
    Returns a dictionary describing accuracy and confidence.

    probs:  array (N, C) of softmax probabilities
    labels: true labels, integer or one-hot
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = _to_class_labels(labels)

    preds = probs.argmax(axis=1)   # predicted class
    conf = probs.max(axis=1)       # confidence = highest probability
    correct = preds == labels

    # Expected Calibration Error (ECE):
    # group predictions by confidence (e.g. 0.9-1.0), and in each group
    # compare the average confidence with the actual accuracy.
    # 0 means confidence matches accuracy exactly; larger means less reliable.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            gap = abs(conf[in_bin].mean() - correct[in_bin].mean())
            ece += in_bin.mean() * gap

    wrong = ~correct
    return {
        "accuracy": correct.mean(),
        "mean_conf": conf.mean(),
        "mean_conf_correct": conf[correct].mean() if correct.any() else np.nan,
        "mean_conf_wrong": conf[wrong].mean() if wrong.any() else np.nan,
        # wrong predictions made with high confidence: the dangerous case,
        # because the collaboration level (RQ3) only shares LOW-confidence inputs
        "wrong_but_confident": (wrong & (conf >= high_conf)).mean(),
        "ece": ece,
    }


def attack_report(clean_probs, adv_probs, labels):
    """
    Attack success rate (ASR): of the inputs the model got RIGHT when clean,
    the fraction it gets WRONG after the attack.
    """
    labels = _to_class_labels(labels)
    clean_ok = np.asarray(clean_probs).argmax(axis=1) == labels
    adv_ok = np.asarray(adv_probs).argmax(axis=1) == labels
    if not clean_ok.any():
        return {"asr": np.nan, "n_clean_correct": 0}
    return {
        "asr": (~adv_ok[clean_ok]).mean(),
        "n_clean_correct": int(clean_ok.sum()),
    }


def print_table(results):
    """results: dict of {name: confidence_report(...)}, printed as a table."""
    cols = ["accuracy", "mean_conf", "mean_conf_correct",
            "mean_conf_wrong", "wrong_but_confident", "ece"]
    print(f"{'setting':<22}" + "".join(f"{c:>20}" for c in cols))
    for name, r in results.items():
        print(f"{name:<22}" + "".join(f"{r[c]:>20.4f}" for c in cols))


if __name__ == "__main__":
    # Quick self-test with fake data, so you can check the script runs.
    rng = np.random.default_rng(0)
    n, c = 1000, 10
    y = rng.integers(0, c, n)

    # fake "clean" model: mostly right and fairly confident
    clean_logits = rng.normal(size=(n, c))
    clean_logits[np.arange(n), y] += 3.0
    clean_probs = np.exp(clean_logits) / np.exp(clean_logits).sum(1, keepdims=True)

    # fake "attacked" outputs: often wrong, sometimes still confident
    adv_logits = rng.normal(size=(n, c))
    adv_logits[np.arange(n), (y + 1) % c] += 2.0
    adv_probs = np.exp(adv_logits) / np.exp(adv_logits).sum(1, keepdims=True)

    print_table({
        "clean": confidence_report(clean_probs, y),
        "attacked": confidence_report(adv_probs, y),
    })
    print("Attack:", attack_report(clean_probs, adv_probs, y))
