# Lop-and-Pore-Detection 🔬🧩

[![Repository](https://img.shields.io/badge/repo-Lop--and--Pore--Detection-blue)](https://github.com/Hemanth7723/Lop-and-Pore-Detection)
[![Languages](https://img.shields.io/github/languages/top/Hemanth7723/Lop-and-Pore-Detection)](https://github.com/Hemanth7723/Lop-and-Pore-Detection)
[![Last commit](https://img.shields.io/github/last-commit/Hemanth7723/Lop-and-Pore-Detection)](https://github.com/Hemanth7723/Lop-and-Pore-Detection/commits/main)

A production-oriented Python project to detect lops and pores (or other defects) in images — includes training pipelines, evaluation, model export, and an interactive Streamlit demo that lets users upload custom images, apply image-enhancement operations, and inspect predictions with visual explanations.

---

## **Table of contents**
- [**Project overview**](#project-overview)
- [**Repository structure**](#repository-structure)
- [**Quick start**](#quick-start)
  - [**Prerequisites**](#prerequisites)
  - [**Run training (example)**](#run-training-example)
  - [**Run Streamlit demo locally**](#run-streamlit-demo-locally)
- [**Dataset**](#dataset)
- [**Data preprocessing & augmentation**](#data-preprocessing--augmentation)
- [**Modeling & training details**](#modeling--training-details)
  - [**Saving & exporting models**](#saving--exporting-models)
- [**Streamlit demo app**](#streamlit-demo-app)
  - [**Image enhancement features inside the app**](#image-enhancement-features-inside-the-app)
  - [**How the app runs inference**](#how-the-app-runs-inference)
  - [**Streamlit UI flow**](#streamlit-ui-flow)
- [**Evaluation & explainability**](#evaluation--explainability)
- [**Deployment**](#deployment)
  - [**Docker example**](#docker-example)
- [**Environment & dependencies**](#environment--dependencies)
- [**Reproducibility & checkpoints**](#reproducibility--checkpoints)
- [**Troubleshooting & FAQ**](#troubleshooting--faq)
- [**Contributing**](#contributing)
- [**Security, privacy & ethical notes**](#security-privacy--ethical-notes)
- [**Acknowledgements**](#acknowledgements)

---

## **Project overview**
This repository demonstrates an end-to-end workflow for building an image-based defect detection pipeline (lops, pores, or similar defects). It includes:
- Data ingestion and preprocessing
- Augmentation recipes appropriate for defect detection
- Model training (PyTorch / TensorFlow-friendly patterns)
- Model export (PyTorch .pt/.pth, TensorFlow .h5/SavedModel, ONNX)
- Interactive Streamlit application for testing custom images and experimenting with preprocessing/enhancement
- Visualization tools (Grad-CAM / saliency) for qualitative inspection

The Streamlit app is a research/demo tool that helps users test model behavior on new images and visually evaluate predictions. It is not a clinical/production certification.

---

## **Repository structure**
A suggested/typical structure — adjust to match actual files in this repo:

- data/
  - raw/                # original images
  - processed/          # resized / normalized images used for training
  - splits/             # train/val/test lists or folders
- notebooks/            # exploratory notebooks (optional)
- app/
  - streamlit_app.py    # Streamlit front-end for custom image testing
- requirements.txt
- README.md

---

## **Quick start**

### **Prerequisites**
- Python 3.8+ recommended
- pip or conda
- GPU recommended for training (NVIDIA + CUDA) but CPU is fine for small tests / Streamlit demo
- Suggested packages: numpy, opencv-python, pillow, albumentations, torch or tensorflow, torchvision (if using PyTorch), scikit-learn, matplotlib, streamlit, onnxruntime (optional)

### **Run training (example)**
1. Clone the repo and set up environment:
```bash
git clone https://github.com/Hemanth7723/Lop-and-Pore-Detection.git
cd Lop-and-Pore-Detection
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

2. Prepare data:
- Place raw images in `data/raw/` and follow dataset split conventions (or run provided helper to create splits).

3. Train:
```bash
# example for PyTorch training
python src/train.py --config configs/train_config.yaml
```
Check `src/train.py` or your training scripts for available CLI options (batch size, learning rate, epochs, resume checkpoint, etc).

### **Run Streamlit demo locally**
1. Install Streamlit requirements (if separate):
```bash
pip install -r app/requirements-streamlit.txt
```
2. Run the app:
```bash
streamlit run app/streamlit_app.py
```
3. Open http://localhost:8501 and upload an image. Use enhancement toggles and run inference.

---

## **Dataset**
This repo does not include a specific public dataset by default. Use your labelled images (defect vs non-defect) organized either as:
- class folders: data/processed/train/defect/*.png, data/processed/train/ok/*.png
- CSV with image path + label

Dataset tips:
- Keep images consistent in format (PNG/JPG) and bit-depth.
- For detection of small defects (pores, lops), ensure high enough resolution or crop to ROIs containing defects.
- If you use an external dataset, add attribution & verify license before sharing.

---

## **Data preprocessing & augmentation**
Key preprocessing steps:
- Resize / crop to target model input size (e.g., 224x224 or 320x320)
- Normalize pixel intensities to match pretraining (ImageNet mean/std if using pretrained backbone)
- Convert grayscale to 3-channel if using ImageNet-pretrained networks (repeat channel or use a 1→3 conversion layer)
- Optionally apply CLAHE / histogram equalization as a preprocessing step (can be toggled in the app)

Recommended augmentation recipes (using albumentations or torchvision):
- Random rotations (±15°)
- Random horizontal/vertical flips (if domain permits)
- Random crops & scale jitter
- Gaussian blur / motion blur (small) — simulate imaging noise
- Random brightness/contrast
- Small elastic transforms (careful for geometry-sensitive defects)

Balance augmentation strength — aggressive transforms can break small-defect signals.

---

## **Modeling & training details**
- Recommended approach: use a robust backbone (ResNet, EfficientNet, MobileNet) + light classification head for patch-level detection or binary classification.
- For small-defect detection consider:
  - Higher-resolution inputs (320x320 or 512x512) if GPU memory permits
  - Two-stage approach: segmentation/localization model (UNet) or patch classifier over sliding windows
- Losses:
  - Binary Cross-Entropy for binary classification
  - Focal Loss for strong class imbalance
  - Weighted loss or oversampling if defects are rare
- Training tips:
  - Stratified splits to preserve label balance
  - Use learning rate scheduler and early stopping on validation metric
  - Save best model by validation AUC / F1

Example PyTorch training snippet (simplified):
```python
# model, optimizer, criterion already set up
for epoch in range(start_epoch, epochs):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # validate...
    torch.save(model.state_dict(), f"models/checkpoints/epoch_{epoch}.pth")
```

### **Saving & exporting models**
Save formats you may want:
- PyTorch: `models/final/model_best.pth` (state_dict or full checkpoint)
```python
torch.save(model.state_dict(), "models/final/model_best.pth")
```
Load:
```python
model = MyModel(...)
model.load_state_dict(torch.load("models/final/model_best.pth", map_location=device))
model.eval()
```

- TensorFlow / Keras: `models/final/model_best.h5` or SavedModel dir
```python
model.save("models/final/model_best.h5")
# load
from tensorflow.keras.models import load_model
model = load_model("models/final/model_best.h5")
```

- ONNX (optional): export for language/framework-agnostic inference
```python
torch.onnx.export(model, dummy_input, "models/final/model_best.onnx", opset_version=12)
```

Document the exact saved-paths in your notebooks or training logs so the Streamlit app can find them.

---

## **Streamlit demo app**
This project includes a Streamlit-based interactive demo that:
- Accepts user-uploaded images or drag-and-drop
- Applies optional image enhancements (see list below)
- Runs model inference on the enhanced image(s)
- Displays prediction probabilities and visual explanation (Grad-CAM heatmap)
- Allows saving / downloading enhanced images + prediction logs

> The Streamlit app is intended for demonstration and human-in-the-loop analysis only — not for automated production decisions.

### **Image enhancement features inside the app**
The app exposes several image-processing options to help visualize defects and potentially improve model inputs:
- Resize & resample controls (preserve aspect or force size)
- Histogram equalization / CLAHE (improves local contrast)
- Brightness / contrast sliders
- Gamma correction slider
- Denoising options (median filter, non-local means)
- Sharpening (unsharp mask)
- Adaptive threshold preview (for binary visual checks)
- Combination presets (e.g., "CLAHE + Denoise" or "Sharpen + Gamma")

These are applied client-side (in Python) using OpenCV, Pillow, or skimage, so the user can experiment with different settings and immediately see model confidence changes.

### **How the app runs inference**
- The app loads the exported model (PyTorch/TensorFlow/ONNX) at startup or on-demand (depending on config).
- Uploaded image is preprocessed exactly as in training:
  - Resize to model input, normalize (same mean/std), convert channels
- Inference:
  - For PyTorch: wrap input in torch.no_grad() and run model to get logits → apply sigmoid/softmax for probabilities
  - For TensorFlow: use model.predict()
- Visual explanation:
  - Optionally compute Grad-CAM or Grad-CAM++ for top predicted class and overlay on enhanced image

Example PyTorch inference helper (inside Streamlit app):
```python
def predict_pytorch(image, model, device):
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.sigmoid(out).cpu().numpy()[0]
    return probs
```

### **Streamlit UI flow**
- Sidebar:
  - Model selection / version
  - Enhancement toggles & sliders
  - Confidence threshold
  - Option to enable Grad-CAM
- Main area:
  - Upload image (single/multiple)
  - Original vs enhanced preview
  - Prediction card with top-k classes and probabilities
  - Grad-CAM overlay and opacity control
  - Download enhanced image + JSON prediction log

---

## **Evaluation & explainability**
- Quantitative metrics to track:
  - Precision / Recall / F1-score (preferred for rare-defect tasks)
  - ROC AUC and PR-AUC
  - Confusion matrix per-class
- Qualitative checks:
  - Visualize model attention (Grad-CAM, integrated gradients) on many examples including false positives & false negatives
  - Check for spurious correlations (e.g., background artifacts triggering detections)
- Keep a simple evaluation script (src/eval.py) that outputs a CSV of per-image predictions and metrics for reproducibility.

---

## **Deployment**

### **Production considerations**
- For public deployment, configure explicit data handling and privacy policies.
- Containerize the Streamlit app and model artifacts for easier deployment (example Dockerfile below).
- Use a model-serving layer for high concurrency (TorchServe, TensorFlow Serving, FastAPI + uvicorn + gunicorn).

### **Docker example**
Example Dockerfile for Streamlit app:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r app/requirements-streamlit.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
```
Build & run:
```bash
docker build -t lop-pore-streamlit .
docker run -p 8501:8501 -v /path/to/models:/app/models lop-pore-streamlit
```

---

## **Environment & dependencies**
Minimum recommended (example):
- Python >= 3.8
- PyTorch >= 1.8 or TensorFlow >= 2.4 (depending on the implementation)
- albumentations
- opencv-python
- pillow
- streamlit
- onnxruntime (if using ONNX)
- scikit-learn, pandas, matplotlib

Example requirements:
- `requirements.txt` — training & dev deps
- `app/requirements-streamlit.txt` — minimal deps for the Streamlit app (streamlit, opencv-python, torch/tensorflow, pillow, numpy)

Consider pinning exact versions for reproducibility.

---

## **Reproducibility & checkpoints**
- Save training logs (CSV / TensorBoard)
- Keep a config file per experiment (YAML/JSON) with hyperparameters and random seeds
- Store model checkpoints in `models/checkpoints/` and final exports in `models/final/`
- Use deterministic flags where possible and record versions of core libraries (torch/tf, numpy)

---

## **Troubleshooting & FAQ**
Q: App is slow on loading models  
A: Load models lazily only when first needed; reduce input resolution for preview; use a smaller backbone.

Q: Predictions inconsistent between training env and Streamlit  
A: Verify preprocessing (resize, normalization, channel ordering) matches exactly.

Q: Grad-CAM overlay not aligning  
A: Ensure Grad-CAM uses the same image scale / resized image passed to the model and that the layer names match.

Q: Too many false positives  
A: Add more negative samples, stronger filtering (confidence threshold), or a post-processing step to remove tiny noisy detections.

---

## **Contributing**
Contributions welcome! Suggested workflow:
1. Fork the repository
2. Create a short-lived branch: `git checkout -b feat/my-change`
3. Add changes, tests or notebooks, and update requirements if needed
4. Open a Pull Request with clear description and reproducible steps

Guidelines:
- Keep notebook outputs minimal in commits
- Document any new model artifacts or dataset schema changes

---

## **Security, privacy & ethical notes**
- Uploaded images in the Streamlit demo are handled by the server running the app. If deployed publicly, inform users whether images are stored or transmitted.
- This tool is for research / visualization purposes only. Do not rely on demo outputs for safety-critical or regulatory use without full validation and domain expert review.

---

## **Acknowledgements**
- Thank you to the dataset providers (please include dataset attribution in the repository when you add a dataset).
- Recommended libraries and resources: PyTorch / TensorFlow, albumentations, OpenCV, Streamlit, Grad-CAM resources.

---
- or scan the repo to replace placeholders (script names, model filenames) with exact names found in your codebase.

Would you like me to create the Streamlit scaffold next?
