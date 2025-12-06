# Medi-GPT: Domain-Adaptive Transformer Implementation

## 🧬 Project Overview
**Medi-GPT** is a first-principles implementation of a Decoder-only Transformer, built entirely in **PyTorch** from scratch (no HuggingFace / High-level APIs). 

The goal of this project was to deconstruct and re-engineer the mathematical foundations of Large Language Models (LLMs) to study how **Multi-Head Self-Attention (MHSA)** converges on specialized medical syntax.

## 🔬 Core Architecture (White-Box Implementation)
Unlike API-wrapper projects, every component here is manually implemented to demonstrate deep architectural understanding:

* **Causal Self-Attention:** Implemented the scaled dot-product attention mechanism with masking to ensure autoregressive generation:  
  $Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V$
* **Residual Streams:** Manual implementation of skip connections (`x + layer(x)`) to preserve gradient flow and mitigate the vanishing gradient problem during backpropagation.
* **Layer Normalization:** Applied Pre-LN architecture (Normalizing before the sub-layer input) for superior training stability.
* **Positional Embeddings:** Learnable vector embeddings to encode sequence order, replacing static sinusoidal encodings.

## 🛠️ Tech Stack
* **Framework:** PyTorch (Low-level `torch.nn` and `torch.functional`)
* **Optimization:** AdamW with Weight Decay
* **Loss Function:** Cross-Entropy Loss
* **Data Pipeline:** Character-level Tokenization on synthetic clinical corpora.

## 📊 Research Goals
1.  **Gradient Flow Analysis:** Observing how residual connections affect loss convergence in deeper networks.
2.  **Domain Adaptation:** Testing how a small-scale transformer learns to cluster medical terminology (e.g., associating "Insulin" with "Diabetes") purely from distributional semantics.

## 🚀 Usage
To initialize the architecture and begin the pre-training loop:

```bash
python main.py
