# 🏔️ Sagarmatha  
### Heterogeneous Ensembling with Hierarchical Task Conditioning (HTC)

Official implementation of **Sagarmatha**, our system for SemEval-2026 Task 9 on multilingual nested taxonomy classification.

---


> Citation will be added after publication.

---

## 🚀 Overview

Sagarmatha addresses **nested multilingual classification** with three hierarchical levels:

- **S1** → Binary detection  
- **S2** → Category classification  
- **S3** → Fine-grained subcategory classification  

A major challenge is **logical consistency across levels**.  
Our solution combines:

1. Heterogeneous multilingual backbone ensembling  
2. Hierarchical Task Conditioning (HTC)  
3. Logical Violation Rate (LVR) metric  

---

## 🧠 Model Architecture

### 1️⃣ Heterogeneous Backbone Ensemble

We fuse multilingual representations from:

- Microsoft **mDeBERTa-v3**
- Google **RemBERT**
- Google Research **LaBSE**
- **mmBERT**
- Meta AI **XLM-RoBERTa**

Each backbone includes:

- Learnable weighted layer pooling  
- Dropout + projection alignment  
- Late fusion via learnable ensemble weights  

---

### 2️⃣ Hierarchical Task Conditioning (HTC)

To enforce structural consistency:

- S3 predictions are conditioned on detached S1 & S2 logits  
- Stop-gradient prevents unstable backward flow  
- Deterministic inference constraints ensure taxonomy validity  

---

### 3️⃣ Logical Violation Rate (LVR)

We introduce a structural consistency metric:

```
LVR = (# invalid hierarchical predictions) / (total predictions)
```

LVR explicitly measures taxonomy violations beyond macro-F1.

---



### Default Training Settings

- Optimizer: AdamW  
- Scheduler: Linear warmup  
- Batch size: 16 
- Max sequence length: 128  
- Epochs: 6  
- Loss: BCEWithLogitsLoss  
- Random seed: 42  

---


## 🔁 Reproducibility

- Framework: PyTorch 2.x  
- Transformers: HuggingFace  
- Hardware: Kaggle T4 (16GB)  
- Seed: 42  

To reproduce leaderboard results:

1. Train R1–R5 backbones  
2. Enable ensemble fusion  
3. Activate HTC head  
4. Apply inference constraints  

---

## 📈 Experimental Highlights

- Reduced Logical Violation Rate  
- Strong macro-F1 across multilingual tracks  
- Competitive leaderboard performance  

---

## ⚖️ Ethical Considerations

This system detects sensitive manifestations in multilingual content.

Deployment recommendations:

- Human-in-the-loop review  
- Cultural context validation  
- Bias monitoring across languages  
- Transparent reporting of failure modes  

Research use only.

---

## 🔮 Future Work

- Multi-seed stability experiments  
- Calibration-aware thresholding  
- Efficient ensemble distillation  
- Cross-cultural robustness analysis  

---

## 📜 License

MIT License (TBD)

---

## 📬 Contact

**Sujal Maharjan**  
[sujalmaharjan007@gmail.com]

---

⭐ If this repository helps your research, please consider starring it.