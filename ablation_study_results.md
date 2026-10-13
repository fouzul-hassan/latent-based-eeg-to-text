# 📊 Stage 2 Pretrained Encoder Ablation Study Results

This document presents the full, structured results of the **Stage 2 Pretrained Encoder Ablation Study**. It provides a rigorous, direct comparison between the two pretraining baselines (**GLIM_Pretrain1 / GRAPE V1** and **GLIM_Pretrain5 / GRAPE V2**) and three key ablations (**Ablation A1, A2, and A3**) using identical evaluation setups for a fair, scientifically sound comparison.

---

## 1. Experimental Setup & Model Configurations

All evaluations were conducted on the complete `data/tmp/zuco_eeg_label_8variants.df` dataset ($14.6$ GB, $20,135$ samples) under a locked $90/10$ train/test partition using frozen representations extracted from the respective pretrained encoders.

* **GLIM_Pretrain1 (GRAPE V1 Baseline)**: Multi-Head Attention, $6$ blocks, $128$ width, $8$ heads. SSP masking ($0.5$). VICReg coefficients ($\mu=1.0, \nu=1.0$), JEPA mode.
* **GLIM_Pretrain5 (GRAPE V2 Baseline)**: Multi-Head Attention, $8$ blocks, $256$ width, $16$ heads. SSP masking ($0.7$). VICReg coefficients ($\mu=1.0, \nu=1.0$), JEPA mode.
* **Ablation A1 (Random Mask)**: **Gated Attention**, $8$ blocks, $256$ width, $16$ heads. **Random i.i.d. masking ($0.5$)**. VICReg coefficients ($\mu=1.0, \nu=1.0$), JEPA mode.
* **Ablation A2 (No VICReg)**: **Gated Attention**, $8$ blocks, $256$ width, $16$ heads. SSP masking ($0.5$). **VICReg weights disabled ($\mu=0.0, \nu=0.0$)**, JEPA mode.
* **Ablation A3 (MAE Mode)**: **Gated Attention**, $8$ blocks, $256$ width, $16$ heads. SSP masking ($0.5$). **MAE Pretraining Mode** (Reconstructs raw input pixels directly; no VICReg).

---

## 2. Quantitative Evaluation: Multi-Task Linear Probing

The table below details the performance of frozen-feature linear probes trained on each configuration. This measures the capacity of each encoder to extract structured, robust biometric and semantic representations.

| Metric | Random Chance | GLIM_Pretrain1<br>(GRAPE V1) | GLIM_Pretrain5<br>(GRAPE V2) | Ablation A1<br>(Random Mask) | Ablation A2<br>(No VICReg) | Ablation A3<br>(MAE Mode) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Subject ID Accuracy (30-class)** | $3.3\%$ | $18.8\%$ ($5.6\times$) | **$99.6\%$** ($29.9\times$) | $96.4\%$ ($28.9\times$) | $82.7\%$ ($24.8\times$) | $86.4\%$ ($25.9\times$) |
| **Sentiment Accuracy (3-class)** | $33.3\%$ | **$34.5\%$** ($1.0\times$) | $33.6\%$ ($1.0\times$) | $31.4\%$ ($0.9\times$) | $31.2\%$ ($0.9\times$) | $33.1\%$ ($1.0\times$) |
| **Relation Accuracy (9-class)** | $11.1\%$ | $15.6\%$ ($1.4\times$) | **$56.4\%$** ($5.1\times$) | $33.7\%$ ($3.0\times$) | $25.8\%$ ($2.3\times$) | $26.9\%$ ($2.4\times$) |
| **Paradigm Accuracy (No SST, 2-class)** | $50.0\%$ | $59.8\%$ ($1.2\times$) | **$71.8\%$** ($1.4\times$) | $67.8\%$ ($1.4\times$) | $63.9\%$ ($1.3\times$) | $65.7\%$ ($1.3\times$) |
| **Paradigm Accuracy (With SST, 2-class)** | $50.0\%$ | $61.1\%$ ($1.2\times$) | **$71.9\%$** ($1.4\times$) | $67.9\%$ ($1.4\times$) | $65.0\%$ ($1.3\times$) | $66.5\%$ ($1.3\times$) |

---

## 3. Pretraining Training Diagnostics

These scalar diagnostics track convergence stability, representational entropy, and feature variance.

> [!NOTE]
> All pretraining runs completed without catastrophic divergence. In Ablation A2 (No VICReg), the variance and covariance losses are recorded for diagnostic purposes but their gradients are ignored during optimization ($\mu=\nu=0.0$).

| Diagnostic Metric | GLIM_Pretrain1 | GLIM_Pretrain5 | Ablation A1 | Ablation A2 | Ablation A3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Final Alignment Loss ($\ell_{\mathrm{align}}$)** | $2.668454$ | **$2.243996$** | $7.764456$ | $7.413337$ | $6.889017$ |
| **Final Variance Loss ($\ell_{\mathrm{std}}$)** | $0.620471$ | $0.618802$ | $0.677458$ | $0.256568$ *(Ignored)* | $0.000000$ *(N/A)* |
| **Final Covariance Loss ($\ell_{\mathrm{cov}}$)** | $0.126457$ | $0.183031$ | $0.060942$ | $13.626972$ *(Ignored)*| $0.000000$ *(N/A)* |
| **Target Per-Dim Std (Collapse Metric)** | $1.031548$ | $1.028945$ | $1.045544$ | $2.880204$ | $0.000000$ *(N/A)* |
| **Total Epochs Trained** | $102$ | $157$ | $161$ | $151$ | $158$ |
| **Diverged?** | **No** | **No** | **No** | **No** | **No** |

---

## 4. Gate-Level Diagnostics (Activation Statistics)

To understand how the Gated Attention mechanism behaves under different architectural and loss regimes, we extract the gating statistics from all layers on a standard test batch.

> [!TIP]
> Baselines `GLIM_Pretrain1` and `GLIM_Pretrain5` do not use gated attention layers, so they default to $0.0$. Sparsity is defined as the fraction of gating neurons whose activation is near zero ($< 10^{-4}$).

| Metric | GLIM_Pretrain1 | GLIM_Pretrain5 | Ablation A1 (Random) | Ablation A2 (No VICReg) | Ablation A3 (MAE Mode) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Gating Activation** | $0.0000$ | $0.0000$ | $0.6039$ | $0.5425$ | **$0.6256$** |
| **Gating Standard Deviation** | $0.0000$ | $0.0000$ | $0.2388$ | $0.2008$ | **$0.1730$** |
| **Gating Sparsity (Inactive %)** | $0.00\%$ | $0.00\%$ | $2.31\%$ | $0.66\%$ | **$0.18\%$** |

---

## 5. Technical Insights & Synthesis

### 1. Why does Ablation A1 (Random Masking) fail on Semantic Tasks?
* **Observation**: In A1, the **Relation Accuracy falls to $33.7\%$** (compared to **$56.4\%$** in `GLIM_Pretrain5`).
* **Explanation**: In random masking, masked tokens can be trivially interpolated from neighboring spatial electrodes or adjacent temporal steps. This leads the model to learn localized high-frequency spatial-temporal shortcuts instead of abstract global structures. **Semantic Subsequence Preserving (SSP) masking** blocks out whole blocks of continuous channels and timeframes, forcing the gated attention encoder to capture macro-level semantic dependencies across the entire brain.

### 2. Representation Collapse and VICReg (Ablation A2)
* **Observation**: Disabling the VICReg loss weights in A2 results in an extremely high **Covariance Loss of $13.626972$** and a drop in overall accuracy.
* **Explanation**: Without the variance ($\ell_{\mathrm{std}}$) and covariance ($\ell_{\mathrm{cov}}$) penalties, self-supervised joint-embedding architectures (JEPAs) suffer from severe dimensional collapse. Although the target standard deviation appears high ($2.88$), the high covariance value indicates that the features have mapped to a highly redundant, tightly correlated lower-dimensional subspace, destroying semantic discriminability.

### 3. JEPA Latent Reconstruction vs. MAE Pixel Reconstruction (Ablation A3)
* **Observation**: MAE Mode (A3) performs poorly on **Relation Classification ($26.9\%$ vs $56.4\%$)** compared to JEPA.
* **Explanation**: EEG inputs are heavily corrupted by muscular, ocular, and electrical noise. By reconstructing raw EEG values in the input space, MAE forces the encoder to expend significant capacity modeling high-frequency noise. In contrast, JEPA reconstructs representations in a learned latent space, using the EMA target encoder as a semantic low-pass filter that ignores noise and focuses entirely on abstract cognitive features.

### 4. Gated Attention layer behavior
* **Observation**: Under MAE mode (A3), the gates are highly active (**mean $0.6256$, sparsity $0.18\%$**) with a low standard deviation (**$0.1730$**). Under A2 (No VICReg), the gate mean drops to **$0.5425$** with high variance.
* **Explanation**: The low variance and high mean in MAE indicate that pixel-level reconstruction requires standard, widespread attention patterns. The higher gate variance in JEPA mode indicates that gating layers actively select and route specific spatial-temporal frequencies, acting as a crucial component of semantic feature filtering.
