# Medi-GPT Lite 🏥

**A PyTorch implementation of a Decoder-only Transformer, built from scratch to study the internal mechanics of Large Language Models within a healthcare context.**

## 📌 Project Overview
Unlike standard API wrappers, this project constructs the mathematical architecture of a GPT-style model manually. It is based on the "nanoGPT" curriculum and *Attention Is All You Need* (Vaswani et al.), adapted to train on a synthetic medical terminology dataset.

The core objective is to demonstrate a "White Box" understanding of:
- **Multi-Head Self-Attention (MHSA):** Implementation of parallel attention heads to capture semantic relationships.
- **Positional Embeddings:** Manual implementation of learnable position vectors.
- **Layer Normalization (Pre-LN):** Applied for training stability in deeper networks.
- **Residual Connections:** Implemented to mitigate vanishing gradients during backpropagation.

## ⚙️ Technical Architecture
* **Framework:** PyTorch
* **Architecture:** GPT-2 style Decoder-only Transformer
* **Optimization:** AdamW Optimizer
* **Loss Function:** Cross-Entropy Loss
* **Tokenization:** Character-level encoding (for architectural transparency)

## 🚀 Key Features
1. **Custom Training Loop:** Implements a full training cycle with batching, forward pass, loss calculation, and backpropagation.
2. **Medical Domain Adaptation:** The model is trained on a specific corpus of chronic condition definitions (Diabetes, Hypertension, Asthma) to observe how embeddings cluster domain-specific terms.
3. **Hyperparameter Config:** structured configuration for flexible experimentation with block size, embedding dimensions, and head counts.

## ⚠️ Limitations
* **Educational Scope:** This model is optimized for CPU training to demonstrate architectural correctness. It is not intended for clinical use.
* **Dataset:** Trained on a small synthetic dataset for proof-of-concept.

## 💻 Usage
To initialize the model, train on the dataset, and generate text:

```bash
python main.py
