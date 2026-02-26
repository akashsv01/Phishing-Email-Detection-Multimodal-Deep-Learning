# Phishing Email Detection with Multimodal Deep Learning

> **ENPM703 — Fundamentals of AI and Deep Learning | Fall 2025**

A dual-tower fusion deep learning system that detects phishing emails by jointly analyzing **email text**, **embedded brand logos**, and **engineered metadata** — achieving **99.45% accuracy** and **AUC 0.999** on a balanced dataset of 76,346 emails.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/anilawork/phish-detection-ui-final)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the Pipeline](#reproducing-the-pipeline)
- [Running Inference](#running-inference)
- [Live Demo](#live-demo)
- [Team](#team)

---

## Overview

Phishing attacks remain one of the most prevalent cyber threats. Traditional text-only filters fail when attackers mimic legitimate brand emails visually. This project addresses that gap with a **multimodal approach** that fuses three complementary signal types:

| Modality | What it captures | Output dim |
|---|---|---|
| **Email Text** | Linguistic patterns, urgency, vocabulary | 256-d |
| **Brand Logo** | Visual brand impersonation via embedded images | 512-d |
| **Metadata** | URL count, capitalization ratio, keyword signals | 64-d (projected from 20) |

The three towers are pre-trained independently as specialists, then fused in a joint classifier that achieves near-perfect detection.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  EMAIL INPUT                        │
                    └──────────┬────────────────┬──────────────┬──────────┘
                               │                │              │
                    ┌──────────▼──────┐  ┌──────▼──────┐  ┌───▼────────────┐
                    │   TEXT TOWER    │  │ IMAGE TOWER │  │ METADATA MLP   │
                    │  (Custom CNN)   │  │ (Custom CNN)│  │  (20-dim feat) │
                    │                 │  │             │  │                │
                    │  Embed(128)     │  │ Conv 3→64   │  │  Linear(20,64) │
                    │  Conv1D ×4      │  │ Conv 64→128 │  │  ReLU          │
                    │  GlobalAvgPool  │  │ Conv128→256 │  │                │
                    │  Linear→256     │  │ Conv256→512 │  └───────┬────────┘
                    └──────────┬──────┘  │ AvgPool→512 │          │
                               │         └──────┬──────┘          │
                               │  256-d         │  512-d          │  64-d
                               └────────────────┴──────────────────┘
                                                │
                                         Concat (832-d)
                                                │
                                    ┌───────────▼───────────┐
                                    │    FUSION CLASSIFIER   │
                                    │  Linear(832→512) + BN  │
                                    │  Linear(512→256) + BN  │
                                    │  Linear(256→128) + BN  │
                                    │  Linear(128→2)         │
                                    └───────────┬────────────┘
                                                │
                                    ┌───────────▼────────────┐
                                    │  Phishing / Legitimate  │
                                    └────────────────────────┘
```

**Training strategy:**
1. **Phase 1** — Text and image towers are trained independently as specialist classifiers.
2. **Phase 2** — Email and logo datasets are aligned into a unified multimodal dataset.
3. **Phase 3** — Towers are loaded with frozen weights; the fusion classifier is trained. Then the full network is fine-tuned end-to-end.

---

## Dataset

### Emails

| Source | Type | Approx. Count |
|---|---|---|
| CEAS_08 | Spam / Phishing | ~17,000 |
| Enron | Legitimate | ~18,000 |
| Nazario | Phishing | ~2,000 |
| Nigerian Prince | Phishing | ~4,000 |
| **Total (balanced)** | 50% phishing / 50% legitimate | **76,346** |

### Brand Logos (Image Tower)

- **Source:** OpenLogo dataset
- **72,652 brand logo images** across **352 brand classes**
- Resized to 224×224 and normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Metadata Features (20-dim vector)

Text length, subject length, body length, URL count, shortened-URL flag, suspicious domain keywords, urgency-word count, action-phrase count, financial-keyword count, capitalization ratio, exclamation-mark count, dollar-sign count, word count, and 7 additional engineered binary/continuous signals.

---

## Results

### Model Performance Comparison

| Model | Accuracy | Notes |
|---|---|---|
| KNN (text features) | 81.71% | Baseline |
| Logistic Regression | 80.00% | Baseline |
| Custom Text CNN | 98.96% | Phase 1 text specialist |
| Custom Image CNN | 76.30% | Phase 1 image specialist |
| ResNet18 (transfer learning) | 97.43% | Comparison baseline |
| **Dual-Tower Fusion** | **99.45%** | **Final model** |

### Fusion Model Detailed Metrics

| Metric | Score |
|---|---|
| Accuracy | 99.45% |
| AUC-ROC | 0.999 |
| Precision (phishing class) | 99.5% |
| Recall (phishing class) | 99.4% |
| F1-Score | 99.4% |

---

## Repository Structure

```
Phishing-Email-Detection-Multimodal-Deep-Learning/
│
├── notebooks/                              # Training notebooks — run in order
│   ├── Final_CNN_Text_1.ipynb                    # Phase 1A: Train text CNN specialist
│   ├── Final_CNN_Images_Custom.ipynb             # Phase 1B: Train image CNN (custom)
│   ├── Final_CNN_Images_Resnet18.ipynb           # Phase 1B alt: ResNet18 comparison
│   ├── dual_tower_text_features.ipynb            # Phase 2: Build unified multimodal dataset
│   ├── train_fusion3.ipynb                       # Phase 3: Train dual-tower fusion model
│   └── baselines/
│       ├── Final_KNN_Text.ipynb                  # KNN baseline
│       └── text_tower_knn.ipynb                  # KNN text tower variant
│
├── src/                                    # Reusable Python modules
│   ├── fusion_models.py          # DualTowerFusionModel, TextFeatureExtractor, ImageFeatureExtractor
│   ├── fusion_dataset_v2.py      # PyTorch Dataset for multimodal training
│   ├── brand_extractor.py        # Extract brand names from email text
│   ├── brand_logo_mapper.py      # Map brand names to logo file paths
│   ├── build_brand_index.py      # Build brand→images JSON index
│   ├── email_ratio.py            # Email dataset balance utilities
│   └── __init__.py
│
├── inference/
│   └── preprocess_html_and_predict.py  # End-to-end inference on raw .html email files
│
├── data/
│   ├── vocab_text_1.json               # Text CNN vocabulary (word→index)
│   ├── class_to_idx_image_custom.json  # Image CNN label map (custom CNN)
│   ├── class_to_idx_image_resnet18.json# Image CNN label map (ResNet18)
│   ├── brand_to_images.json            # Brand→logo file paths index
│   ├── cleaned_combined_emails.csv     # Preprocessed email dataset           [git-lfs]
│   └── unified_multimodal_text.csv     # Unified multimodal training dataset  [git-lfs]
│
├── models/
│   ├── best_custom_cnn_text_1.pth      # Text specialist weights  (~104 MB)   [git-lfs]
│   ├── best_custom_cnn_image_custom.pth# Image specialist weights (~31 MB)    [git-lfs]
│   └── best_fusion_model.pth           # Final fusion model weights (~137 MB) [git-lfs]
│
├── docs/
│   ├── Project_Report.pdf              # Full project report
│   ├── Architecture_Report.pdf         # System architecture report
│   ├── Presentation.pdf                # Project presentation slides
│   ├── Contribution_Report.pdf         # Team contribution report
│   ├── Presentation_Recording.mp4      # Presentation recording
│   └── Demo_Recording.mp4              # Live demo recording
│
├── requirements.txt
├── .gitattributes                      # git-lfs tracking rules
├── .gitignore
└── README.md
```

> **Files marked `[git-lfs]`** are tracked with Git Large File Storage. Run `git lfs pull` after cloning to download them.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended; CPU inference is supported but slower)
- [Git LFS](https://git-lfs.com/) for model weights and large datasets

### Clone and Install

```bash
# 1. Install Git LFS (if not already installed)
git lfs install

# 2. Clone — LFS files download automatically
git clone https://github.com/akashsv01/Phishing-Email-Detection-Multimodal-Deep-Learning.git
cd Phishing-Email-Detection-Multimodal-Deep-Learning

# 3. If LFS files did not download automatically
git lfs pull

# 4. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 5. Install dependencies
pip install -r requirements.txt
```

---

## Reproducing the Pipeline

Run the notebooks in order. Each phase produces artifacts consumed by the next.

---

### Phase 1 — Specialist Pre-training

#### 1A. Text Specialist CNN

**Notebook:** `notebooks/Final_CNN_Text_1.ipynb`

- **Input:** `data/cleaned_combined_emails.csv`
- Builds a character-level vocabulary and trains a 4-block 1D CNN on tokenized email text
- **Outputs:** `models/best_custom_cnn_text_1.pth`, `data/vocab_text_1.json`
- **Result:** 98.96% accuracy

#### 1B. Image Specialist CNN

**Notebook:** `notebooks/Final_CNN_Images_Custom.ipynb`

- **Input:** OpenLogo brand logo dataset (download separately — see note below)
- Trains a VGG-style 4-block 2D CNN on 224×224 logo images
- **Outputs:** `models/best_custom_cnn_image_custom.pth`, `data/class_to_idx_image_custom.json`
- **Result:** 76.30% accuracy (352-class logo classification)

> **OpenLogo dataset:** Not included due to size (~2 GB). Download from [qmul-openlogo.github.io](https://qmul-openlogo.github.io/) and set the path in the notebook.

**Comparison baseline:** `notebooks/Final_CNN_Images_Resnet18.ipynb` — uses ResNet18 transfer learning (97.43% accuracy on the email classification task).

---

### Phase 2 — Multimodal Data Integration

**Notebook:** `notebooks/dual_tower_text_features.ipynb`

- Extracts brand names from each email using `src/brand_extractor.py`
- Maps each email to its most relevant brand logo file path via `data/brand_to_images.json`
- Merges text features with image paths and metadata into a single aligned dataset
- **Output:** `data/unified_multimodal_text.csv`

---

### Phase 3 — Fusion Model Training

**Notebook:** `notebooks/train_fusion3.ipynb`

- Loads pre-trained text and image tower weights from Phase 1
- **Stage 1:** Freezes both towers; trains only the fusion classifier and metadata MLP
- **Stage 2:** Unfreezes towers; fine-tunes the full network end-to-end
- **Output:** `models/best_fusion_model.pth`
- **Result:** 99.45% accuracy, AUC 0.999

---

## Running Inference

Classify a raw `.html` email file using the trained fusion model:

```bash
python inference/preprocess_html_and_predict.py path/to/email.html
```

**Pipeline (fully automatic):**

1. Parse HTML → extract subject, body text, and embedded/linked images
2. Tokenize text using `data/vocab_text_1.json`
3. Decode and resize the first image to 224×224
4. Extract 20 metadata features (URL patterns, keyword signals, character statistics)
5. Run all three tensors through the fusion model
6. Print verdict + confidence + detected suspicious signals
7. Save a `.txt` report alongside the input file

**Example output:**

```
======================================================================
Analyzing: suspicious_email.html
======================================================================

Step 1: Parsing HTML...
   Subject: Your account has been suspended - Immediate action required...
   Body length: 2847 chars
   Images found: 2

Step 2: Processing text...
Step 3: Processing image...
Step 4: Extracting metadata...
Step 5: Running fusion model...

======================================================================
ANALYSIS RESULTS
======================================================================

Prediction: PHISHING
Confidence: 98.73%

Detected Issues:
  Contains shortened URLs (bit.ly, tinyurl)
  High urgency language (5 urgent keywords)
  Multiple call-to-action phrases
  Excessive capitalization (34.2%)
======================================================================
```

---

## Live Demo

A Streamlit application is deployed on Hugging Face Spaces — no installation required:

**[https://huggingface.co/spaces/anilawork/phish-detection-ui-final](https://huggingface.co/spaces/anilawork/phish-detection-ui-final)**

Upload any `.html` email file to receive an instant phishing verdict with confidence score and suspicious signal breakdown.

---

## Team

- [Vishal Patil](https://github.com/VishalPatil18)
- [Akash S Vora](https://github.com/akashsv01)
- [Srihari Narayan](https://github.com/Srihari-Narayan)
- [Sai Anila Namburi](https://github.com/madhu-anila)
---

## References

- OpenLogo Dataset — Queen Mary University of London  
  https://qmul-openlogo.github.io/
- CEAS 2008 Spam Filtering Challenge  
- Phishing Email Dataset — Naser Abdullah Alam  
  https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- PyTorch — https://pytorch.org
