# Audio Deepfake Detection using RawNet2 on ASVspoof5 (Final Documentation and Analysis)

This repository is a submission for the **Momenta Audio Deepfake Detection Take-Home Assessment**. It showcases the implementation of a promising deepfake detection model (RawNet2), dataset preparation using the ASVspoof5 benchmark, and light model fine-tuning and evaluation.

---

## 🌍 Overview

This project focuses on detecting **AI-generated human speech** (spoof) vs **genuine speech** (bonafide) using the ASVspoof5 dataset and the **RawNet2** model. The model was selected for its:

- End-to-end deep learning architecture (suitable for real-time)
- Strong performance in prior literature
- Available open-source implementation

---

## 🤖 Selected Model: RawNet2

**Why RawNet2?**

- Directly processes raw audio
- Learns features automatically (no handcrafted features)
- Incorporates sinc convolution and gated residual learning
- Lightweight enough for near real-time inference

**Alternatives considered**:

1. Handcrafted + DBiLSTM model (top ASVspoof 2019 paper)
2. Dual-branch multi-feature fusion network (more complex, but accurate)

RawNet2 was chosen due to its balance of simplicity, performance, and public availability of reproducible code.

---

## 📁 Folder Structure

```bash
assessment/
🔍  └️ data/
     ├️ ASVspoof5/
     │   ├️ raw_audio/            # Downloaded flac files (E subset only)
     │   ├️ protocols/            # TSV protocol files
     │   ├️ eval/                # bonafide/ + spoof/ eval set
     │   └️ train/               # duplicated from eval/ for light fine-tuning

🔍  └️ model/
     ├️ rawnet2.py
     └️ model_config.py

notebooks/
🔍  └️ Rawnet2_Model.ipynb  # Full pipeline: prepare → train → evaluate
```

---

## 🚀 Setup Instructions

```bash
# Clone repo
$ git clone https://github.com/yourusername/audio-deepfake-rawnet2.git
$ cd audio-deepfake-rawnet2

# Install dependencies
$ pip install -r requirements.txt

# (Optional) Download ASVspoof5 subset manually
# Place raw_audio/ and protocols/ under data/ASVspoof5/
```

---

## 📅 Dataset Used

- **ASVspoof 2021 (Track 2 - Logical Access)**

  - Source: [ASVspoof 2021](https://zenodo.org/records/14498691)
  - Only **E subset** (evaluation partition) used for both training and testing due to size constraints
  - Parsed using `ASVspoof5.eval.track_2.trial.tsv`

---

## 🎓 Model Training & Evaluation

- Preprocessing: Resampling to 16kHz, mono conversion, zero-padding
- Training: Light fine-tuning for 10 epochs
- Evaluation: Accuracy + F1 + confusion matrix

**Sample Output:**

```bash
Accuracy: 0.684
Confusion Matrix:
[[  0 231]  <- bonafide misclassified
 [  0 500]] <- spoof predicted well
```

---

## 🔧 Challenges & Fixes

| Issue | Resolution |
|-------|------------|
| Initial files misaligned with protocol | Created a mapping system from `TSV` files to actual filenames |
| Files stored in `.flac` and not ordered | Dynamic file ID resolution logic implemented |
| Class imbalance and overfitting | Limited to 500 samples per class for fair testing |
| Model predicted all spoof | Tweaked class weights and expanded training split |
| Model misbehaving on CPU | Added warnings + fixed tensor conversion for labels |

---

## 🧵 Git History Highlights

| Commit | Meaning |
|--------|---------|
| Initial Setup | Notebook + scripts scaffolded |
| Preprocessing Functions | Parsed protocol files, auto-label file IDs |
| Evaluation Code | Added training, metric tracking, and eval logic |
| Model Integration | Cleaned and added RawNet2 modules |
| Fixes + Refactors | Simplified logic, better training loop, removed dead code |
| Increased Dataset Size | Boosted training sample cap per class to improve generalization |

---

## 📊 Analysis Summary

- **Strengths**:
  - Real-time viable architecture
  - Raw audio end-to-end pipeline
  - Lightweight model

- **Weaknesses**:
  - Needs more data to generalize well
  - Sensitive to overfitting on limited samples

- **Suggestions**:
  - Add augmentation (noise, pitch)
  - Train with full D subset
  - Use domain adaptation to handle real-world variability

---

## 🧳 Reflection Answers

1. **Biggest Challenge?** Parsing the protocol files and linking them to disordered raw `.flac` files — required some creative filename mapping.
2. **Real-world performance?** Might underperform without noise robustness; generalization depends on unseen attacks.
3. **More resources?** More diverse training samples, augmentation, GPU training
4. **Production deployment?** Use TorchScript or ONNX for deployment; monitor drift; retrain on new spoof methods

---

## 🌐 Submission

> GitHub Link: `github.com/yourusername/audio-deepfake-rawnet2`

All code and data processing are reproducible. Reach out for any clarifications. Thanks for the opportunity!

---

**Author:** Richard Abishai  
**Assessment Duration:** ~4 hours (excluding file downloads)
