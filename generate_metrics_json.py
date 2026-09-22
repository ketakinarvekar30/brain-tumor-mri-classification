"""
Generates outputs/metrics.json for the CortexScan dashboard's Metrics tab.

This reuses the same evaluation logic as your evaluate.py, but saves the
results as JSON (instead of only printing/plotting) so the frontend can
display them.

Run this AFTER evaluate.py has been run at least once (or run standalone,
it re-runs the test set itself):

    python generate_metrics_json.py
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

from config import DEVICE, BEST_MODEL
from models.cnn import BrainCNN
from utils.dataset import get_dataloaders

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_COLORS = {
    "glioma": "#d84a3c",
    "meningioma": "#c07a17",
    "notumor": "#0f9e88",
    "pituitary": "#4a4fe0",
}


@torch.no_grad()
def main():
    _, _, test_loader, _ = get_dataloaders()

    model = BrainCNN().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = F.softmax(outputs, dim=1).argmax(dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    per_class = []
    for name in CLASS_NAMES:
        c = report[name]
        per_class.append({
            "name": name.capitalize() if name != "notumor" else "No Tumor",
            "precision": round(c["precision"], 4),
            "recall": round(c["recall"], 4),
            "f1": round(c["f1-score"], 4),
            "support": int(c["support"]),
            "color": CLASS_COLORS[name],
        })

    output = {
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": per_class,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved outputs/metrics.json")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()