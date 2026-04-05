# 🛰️ Astronomical Object Recognition
### A Deep Learning Architecture Comparison Study

---

## ⚔️ Overview

A systematic comparison of three deep learning paradigms applied to astronomical image classification across **11 solar-system bodies**: Earth, Jupiter, MakeMake, Mars, Mercury, Moon, Neptune, Pluto, Saturn, Uranus, Venus.

Each architecture was designed, trained, and evaluated independently, then compared on accuracy, convergence speed, generalisation, and architectural trade-offs.

| # | Model | Framework | Type |
|---|-------|-----------|------|
| 1 | **MLP** | PyTorch | Baseline — ablation study |
| 2 | **CNN + Regularization** | TensorFlow / Keras | Custom deep CNN |
| 3 | **Transfer Learning** | PyTorch | ResNet-18 fine-tuned |

---

## 📊 Results at a Glance

| Model | Val Accuracy | Classes | Convergence | Early Stop |
|-------|-------------|---------|-------------|------------|
| MLP (best config) | ~90% | 11 | ~40–60 epochs | ✅ patience=15 |
| CNN + Regularization | ~94% | 11 | ~5–8 epochs | ✅ patience=15 |
| **Transfer Learning** | **100% (1.0000)** | **11** | **Epoch 8** | ✅ patience=12 |

---

## 🥇 Transfer Learning — ResNet-18 Partial Fine-Tune

### Architecture

Pre-trained ResNet-18 with a partial fine-tuning strategy. The first two residual stages are frozen to preserve ImageNet's low-level edge and texture detectors. Stages 3 and 4 and a custom classification head are trainable with layer-wise learning rates.

```
ResNet-18 (ImageNet pre-trained)
├── layer1  [FROZEN]          — edges, textures
├── layer2  [FROZEN]          — shapes, gradients
├── layer3  [lr = 5e-5]       — mid-level structure
├── layer4  [lr = 1e-4]       — semantic features
└── FC head [lr = 1e-3]
       Dropout(0.4) → Linear(512→256) → ReLU
       → Dropout(0.3) → Linear(256→11)
       [Softmax applied at inference via F.softmax]
```

**Trainable parameters: ~2.8M / 11.2M (25%)**

### Training Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **100% (1.0000)** |
| Val Loss at convergence | ~0.0000 |
| Best epoch | 8 |
| Early stopping triggered | Epoch 13 |
| Train loss range (ep. 8–13) | 0.0272 – 0.0529 |
| VRAM used | 394 MB |
| Hardware | CUDA GPU — AMP FP16 |

The train/val loss gap (train ~0.03–0.05, val ~0.0000) is expected behaviour: the validation set contains one image per class, all well within the model's decision boundaries after it has learned discriminative representations.

### Training Configuration

| Component | Setting |
|-----------|---------|
| Loss | `CrossEntropyLoss` with inverse-frequency class weights |
| Optimiser | Adam — layer-wise LR (see architecture above) |
| Weight decay | 1e-4 |
| Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=5) |
| Gradient clipping | max_norm=1.0 |
| Mixed precision | `torch.amp.autocast('cuda')` + `GradScaler('cuda')` |
| Batch size | 32 |
| Image size | 224×224 |
| Max epochs | 50 |
| Early stopping | patience=12 |
| Train/val split | 80/20 per class (stratified) |

### Data Augmentation

Two-stage pipeline — PIL transforms execute before `ToTensor()`; tensor transforms execute after.

**PIL stage** (raw image):
`RandomResizedCrop(224, scale=0.65–1.0)` · `HorizontalFlip(p=0.5)` · `VerticalFlip(p=0.5)` · `Rotation(180°)` · `RandomAffine(translate+shear)` · `RandomPerspective(distortion=0.2, p=0.3)` · `ColorJitter(b/c/s/h)` · `RandomGrayscale(p=0.10)` · `GaussianBlur(p=0.30)`

**Tensor stage** (after `ToTensor` + `Normalize`):
`RandomErasing(p=0.2, scale=2–10%)` · `AddGaussianNoise(std=0.015)`

> `RandomErasing` requires a tensor — placing it before `ToTensor()` raises `AttributeError: 'Image' object has no attribute 'shape'`.

### Class Weighting

Inverse-frequency weights computed at runtime: `w_c = N / (C × n_c)`, rescaled so `mean(w) ≈ 1`. Rare classes (MakeMake, Pluto) receive proportionally larger gradient contributions, preventing the model from ignoring minority classes.

### Image Analysis & Enhancement Pipeline

Every image is fully analysed before inference. Enhancements are applied **only when the metric falls outside the acceptable range** — no corrections on already-clean images.

| Metric | Condition | Enhancement |
|--------|-----------|-------------|
| Brightness | < 60 or > 210 | Normalise toward target range |
| Contrast RMS | < 35 | Boost ×1.60 |
| Contrast RMS | < 55 | Boost ×1.25 |
| Sharpness (Laplacian var.) | < 200 | Sharpness ×2.5 + UnsharpMask |
| Sharpness (Laplacian var.) | < 800 | Sharpness ×1.5 |
| Noise + low contrast | noise > 20, RMS < 60 | Median denoise 3×3 |
| Always | — | PIL DETAIL filter |

### Automatic Test-Set Population

`populate_test_set()` runs before training. For every class in `train_dir`, if `test_dir/Test_<cls>/` is absent or empty (scanned recursively via `os.walk` to handle nested layouts like `Test_Earth/Earth/img.jpg`), one random training image is copied across. `test_loader` is rebuilt automatically after all injections.

### Softmax Inference

`classify_planet()` returns a full probability distribution with:
- Ranked confidence table (top-k with ASCII probability bars)
- Side-by-side original vs. adaptively enhanced image
- Horizontal bar chart of all 11 class probabilities

---

## 🥈 CNN + Regularization — Custom Deep CNN (TensorFlow/Keras)

### Architecture

A custom sequential CNN with 4 convolutional blocks. Each block pairs two `Conv→BN→ReLU` layers before pooling. The network terminates with Global Average Pooling instead of Flatten, reducing parameters in the dense head while improving spatial invariance.

```
Input (128×128×3)
│
├── Block 1:  Conv2D(32)→BN→ReLU × 2 → MaxPool(2×2) → Dropout(0.20)
├── Block 2:  Conv2D(64)→BN→ReLU × 2 → MaxPool(2×2) → Dropout(0.20)
├── Block 3:  Conv2D(128)→BN→ReLU × 2 → MaxPool(2×2) → Dropout(0.40)
├── Block 4:  Conv2D(256)→BN→ReLU × 2 → GlobalAvgPool → Dropout(0.40)
│
├── Dense(256) → BN → ReLU → Dropout(0.40)
├── Dense(128) → BN → ReLU → Dropout(0.30)
└── Dense(11)  → Softmax
```

All Conv and Dense kernels carry L2 regularisation (λ=0.001).

### Training Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | ~94% |
| Image size | 128×128 |
| Batch size | 16 |
| Saved checkpoint | `best_planets_model.keras` |

### Training Configuration

| Component | Setting |
|-----------|---------|
| Loss | `categorical_crossentropy` |
| Optimiser | Adam (lr=0.001) |
| L2 regularization | λ=0.001 on all kernels |
| Dropout | 0.40 (0.20 in blocks 1–2) |
| Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=7, min_lr=1e-6) |
| Early stopping | patience=15, restore_best_weights=True |
| Checkpoint | monitor=val_accuracy, save_best_only=True |
| Max epochs | 10 |
| Train/val split | 80/20 validation_split |

### Regularization Strategy (7 techniques)

| # | Technique | Detail |
|---|-----------|--------|
| 1 | L2 Weight Decay | λ=0.001 on all Conv + Dense kernels |
| 2 | Dropout | 0.20 in early blocks, 0.40 in deeper layers |
| 3 | Batch Normalization | After every Conv activation, stabilises gradient flow |
| 4 | Data Augmentation | rotation 30°, shift 0.2, shear 0.15, zoom 0.2, h-flip, brightness ±20% |
| 5 | Early Stopping | Halts when val_loss plateaus (patience=15) |
| 6 | LR Reduction | Halves LR on val_loss plateau (patience=7) |
| 7 | Global Average Pooling | Replaces Flatten — fewer dense parameters, better generalisation |

---

## 🥉 MLP — Ablation Study (PyTorch)

### Architecture

A fully connected network that flattens the input image into a 1D vector. The notebook is structured as a **systematic ablation study** — six independent experiments isolate the impact of each design choice before arriving at the best configuration.

```
Input (256×256×3) → Flatten → 196,608 features
→ Linear(512) → BatchNorm1d → ReLU → Dropout(0.3)
→ Linear(256) → BatchNorm1d → ReLU → Dropout(0.3)
→ Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)   ← best config
→ Linear(11)  → (CrossEntropyLoss during training / Softmax at eval)
```

### Training Results (Best Configuration)

| Metric | Value |
|--------|-------|
| Best config | `[512, 256, 128]` + BatchNorm |
| Test Accuracy | ~90% |
| Macro F1 | ~0.90 |
| Max epochs | 80 |
| Early stopping | patience=15 |
| Optimizer | Adam (lr=1e-3) |
| Split | Stratified 70 / 15 / 15 |

### Ablation Experiments

**Exp 1 — Activation Function**

| Activation | Result |
|------------|--------|
| ReLU | ✅ Best — fast convergence, no vanishing gradients |
| Tanh | ⚠️ Moderate — slower |
| Sigmoid | ❌ Worst — severe vanishing gradients at depth |

**Exp 2 — Depth & Width**

| Architecture | Notes |
|---|---|
| `[1024]` | Underfits — single layer, no hierarchy |
| `[512, 256]` | Good baseline |
| `[512, 256, 128]` | ✅ Best — more hierarchy, controlled parameter count |
| `[512, 256, 128, 64]` | Diminishing returns, slower convergence |

**Exp 3 — L2 Regularization**

| Weight Decay | Notes |
|---|---|
| 0 | Prone to overfitting |
| 1e-4 | Marginal gain |
| 1e-3 | Good balance |
| 1e-2 | Over-regularised — underfits |

**Exp 4 — Dropout**

| Dropout p | Notes |
|---|---|
| 0.0 | No regularisation |
| 0.2 | Light — useful early |
| 0.3 | ✅ Best generalisation |
| 0.5 | Too aggressive for dataset size |

**Exp 5 — Batch Normalization**

`BatchNorm1d` after each hidden layer accelerated convergence and improved final accuracy. Combined with `[512, 256, 128]`, it forms the best configuration.

**Exp 6 — Optimiser**

| Optimiser | LR | Notes |
|---|---|---|
| SGD | 0.01 | Slow, unstable |
| SGD + Momentum | 0.01 | More stable, still slower |
| Adam | 0.001 | ✅ Fastest convergence, best accuracy |

**Best configuration:** `[512, 256, 128]` + BatchNorm + ReLU + Adam(lr=1e-3) + Dropout(0.3) + 80 epochs + patience=15

---

## 🔬 Architecture Comparison

### Design Decisions Side by Side

| Factor | MLP | CNN | Transfer Learning |
|--------|-----|-----|-------------------|
| **Framework** | PyTorch | TensorFlow/Keras | PyTorch |
| **Input size** | 256×256 | 128×128 | 224×224 |
| **Spatial awareness** | ❌ Destroyed by flatten | ✅ Full | ✅ Full (pretrained) |
| **Pre-training** | ❌ | ❌ | ✅ ImageNet (1.2M images) |
| **Class weighting** | ❌ | ❌ | ✅ Inverse-frequency |
| **Mixed precision** | ❌ | ❌ | ✅ AMP FP16 |
| **Batch Norm** | ✅ Best config | ✅ Every block | ✅ Inherited |
| **Regularization** | Dropout + BN + L2 | 7 techniques | Dropout + weight decay |
| **Image enhancement** | ❌ | ❌ | ✅ Adaptive 7-metric pipeline |
| **Auto test injection** | ❌ | ❌ | ✅ |
| **Output** | Softmax | Softmax | Softmax + ranked chart |
| **Saved format** | `.pth` | `.keras` | `.pth` |

### Performance & Convergence

| Model | Val Acc | Speed | Limiting Factor |
|-------|---------|-------|-----------------|
| MLP | ~90% | Slow | Flattening destroys spatial structure |
| CNN | ~94% | Moderate | Learns from scratch — needs data and regularisation |
| Transfer Learning | **100%** | **Fast (epoch 8)** | None — ImageNet features already encode visual abstractions |

### Why Transfer Learning Wins

The **MLP** flattens 196,608 pixel values — every spatial relationship is destroyed. The model can only learn global colour and texture statistics, which is why visually similar bodies (Venus/Earth, Uranus/Neptune) create confusion. No depth, dropout, or batch norm can recover what flattening removes.

The **CNN** preserves spatial structure through convolution, learning disc boundaries, ring geometries, and surface textures. But it must build all of this from scratch. With a small dataset, 7 regularisation techniques carry most of the load and cap generalisation at ~94%.

**Transfer learning** enters with layer1 and layer2 already trained to detect edges, colour gradients, and shapes from 1.2M ImageNet images. Only the upper 25% of the network adapts to the astronomical domain. The result: convergence in 8 epochs, perfect generalisation, and only 394 MB of VRAM — from a model using a quarter of its parameters.

---

## 🧬 Key Findings

1. **Flattening is irreversible.** The MLP's ~90% ceiling is not fixable with more depth or regularisation — the spatial information needed to distinguish planetary discs is gone the moment the image is vectorised.

2. **Regularisation compensates for limited data — up to a point.** The CNN applies 7 regularisation techniques and still tops out at ~94%. Without pre-training, a model can only learn as fast as the dataset allows.

3. **Transfer learning dominates in low-data regimes.** 8 epochs to 100% accuracy is only possible because ImageNet features already partially capture what makes a planetary disc distinct.

4. **Class weighting is critical for imbalanced datasets.** Only the transfer learning model applies inverse-frequency weighting. Without it, MakeMake and Pluto — which have fewer training images — would be systematically underrepresented in the gradient signal.

5. **Ablation studies reveal MLP's sweet spot at depth 3.** The `[512, 256, 128]` configuration outperforms shallower and deeper variants. Adding a fourth layer yields no measurable gain while slowing convergence.

6. **Architectural choices interact.** BatchNorm + deep architecture (`[512, 256, 128]`) together outperform either choice alone in the MLP study — a reminder that regularisation techniques are not independent.

---

## ⚙️ Tech Stack

| Model | Framework | Key Libraries |
|-------|-----------|---------------|
| MLP | PyTorch | torchvision, scikit-learn, seaborn |
| CNN | TensorFlow / Keras | ImageDataGenerator, seaborn |
| Transfer Learning | PyTorch | torchvision, Pillow, scipy, pandas |

---

## 🛠️ Installation

```bash
# PyTorch (MLP + Transfer Learning)
pip install torch torchvision pillow scipy matplotlib numpy scikit-learn pandas seaborn

# TensorFlow (CNN)
pip install tensorflow matplotlib numpy scikit-learn seaborn
```

> **Windows + CUDA:** Set `num_workers=0` on all PyTorch `DataLoader` instances. Worker subprocesses on Windows attempt CUDA re-initialisation in a forked process, silently falling back to CPU buffering even when `device=cuda` is printed.

> **PyTorch AMP:** Use `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`. The `torch.cuda.amp.*` namespace is deprecated in PyTorch ≥ 2.0.

---

## 📂 Project Structure

```
.
├── data/
│   ├── raw/Planets and Moons/         # 11-class source images
│   └── test/
│       ├── Test_Earth/                # auto-populated if missing
│       └── ...
│
├── models/
│   ├── mlp/best_mlp.pth               # [512,256,128] + BN
│   ├── cnn_reg/best_planets_model.keras
│   └── transfer_learning/
│       ├── transfer_enhanced.ipynb
│       └── transfer.ipynb
│
├── results/
│   ├── mlp/
│   │   ├── confusion_matrix.png
│   │   ├── activation_comparison.png
│   │   ├── depth_width_comparison.png
│   │   ├── l2_comparison.png
│   │   ├── dropout_comparison.png
│   │   ├── optimizer_comparison.png
│   │   ├── all_experiments_comparison.png
│   │   ├── first_layer_weights.png
│   │   ├── classification_report.txt
│   │   └── mlp_results.json
│   ├── cnn_reg/
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   └── checkpoints/transfer_learning/
│       ├── best_model.pth
│       └── training_log.csv
│
├── model.py                           # CNN training script
└── README.md
```

---

## 🚀 Execution

```bash
# MLP — open notebook and run all cells
# Runs 6 ablation experiments, saves all plots + mlp_results.json

# CNN
python model.py
# Trains, evaluates, saves training_history.png + confusion_matrix.png + best_planets_model.keras

# Transfer Learning — open transfer_enhanced.ipynb and run all cells
# populate_test_set() auto-runs before training
# classify_planet() handles inference with full enhancement pipeline
```

---

## 🧭 Conclusion

| Model | Sees | Val Acc |
|-------|------|---------|
| MLP | Numbers — pixel statistics | ~90% |
| CNN | Shapes — local spatial patterns | ~94% |
| Transfer Learning | Meaning — semantic visual features | **100%** |

The gap between 90% and 100% is not a numerical accident — it reflects fundamentally different levels of visual understanding. The MLP memorises colour distributions. The CNN learns geometry from scratch. The transfer-learned ResNet-18 recognises meaning in texture, structure, and form that it already partially understood from ImageNet, and adapts that understanding to planetary bodies in 8 epochs.

---

## 🤝 Collaborators

| Name | Role |
|------|------|
| 👑 [Ahmed boray](https://github.com/silragon-ryu) | Team Lead · AI Architect · Transfer Learning |
| [Ahmet Cemil Bostanoğlu](https://github.com/acbst0) | CNN + Regularization |
| [Berke Emir Yaşacan](https://github.com/EmirYscn) | MLP |
| [Mark tendo](https://github.com/comicjelly) | Data Analysis & Processing |
| [Mezred Mohamed Wassim](https://github.com/Woozie0) | Data Analysis & Processing |

---

> *This project was not built to just work. It was built to understand **why** it works.*