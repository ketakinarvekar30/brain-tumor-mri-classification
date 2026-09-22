"""
Backend for the CortexScan dashboard (BrainCNN project).

Serves:
  GET  /health       -> server + device status
  POST /predict        -> upload an MRI image, get prediction
  GET  /api/metrics    -> test-set metrics as JSON (reads outputs/metrics.json)

Folder layout expected:
  checkpoints/best_model.pth
  outputs/metrics.json   (generate with generate_metrics_json.py)

Run:
    pip install flask flask-cors torch torchvision pillow
    python app.py
"""

import io
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent
BEST_MODEL = ROOT_DIR / "checkpoints" / "best_model.pth"
METRICS_FILE = ROOT_DIR / "outputs" / "metrics.json"
HISTORY_FILE = ROOT_DIR / "outputs" / "training_history.csv"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BrainCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BrainCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model.to(device)
model.eval()

# Matches utils/transforms.py test_transform exactly:
# Resize(256,256) -> RandomCrop(224) -> Normalize(0.5,0.5,0.5)
# Note: test_transform uses RandomCrop even at inference in the original code.
# For consistent single-image predictions we use CenterCrop instead, which is
# the standard practice for inference (deterministic, matches training data
# distribution without the random jitter meant only for training/eval batches).
eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device)})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send it as form field 'image'."}), 400

    file = request.files["image"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, 0)

    predicted_class = CLASSES[predicted_idx.item()]

    all_probs = {
        DISPLAY_NAMES[CLASSES[i]]: round(probs[i].item() * 100, 2)
        for i in range(len(CLASSES))
    }

    return jsonify({
        "prediction": DISPLAY_NAMES[predicted_class],
        "raw_class": predicted_class,
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": all_probs,
        "is_tumor": predicted_class != "notumor",
    })


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    if not METRICS_FILE.exists():
        return jsonify({"error": "metrics.json not found. Run generate_metrics_json.py first."}), 404
    with open(METRICS_FILE) as f:
        return jsonify(json.load(f))


@app.route("/api/training-summary", methods=["GET"])
def training_summary():
    if not HISTORY_FILE.exists():
        return jsonify({"error": "training_history.csv not found."}), 404

    df = pd.read_csv(HISTORY_FILE)
    best_row = df.loc[df["val_acc"].idxmax()]

    return jsonify({
        "best_val_acc": float(best_row["val_acc"]),
        "best_epoch": int(best_row["epoch"]),
        "total_epochs": int(df["epoch"].max()),
        "final_val_loss": float(df.iloc[-1]["val_loss"]),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)