# Narrative Frame Prediction via Multimodal Deep Learning

## Quick Links
- **[Experiments Notebook](experiments.ipynb)** – Full training and evaluation workflow
- **[XAI Figures](results/xai/)** – Integrated Gradients, Grad-CAM++, Attention Scores
- **[Loss Curves](results/loss_curves.png)** – Training progress over 25 epochs
- **[Prediction Example](results/prediction_example.png)** – Qualitative story continuation

---

## Innovation Summary

This project tackles **narrative frame prediction**: given K=4 sequential image-caption pairs
from the [StoryReasoning dataset](https://arxiv.org/abs/2505.10292), predict the next story frame.

| # | Component | Baseline | Innovation | Justification |
|---|-----------|----------|------------|---------------|
| 1 | **Fusion** | Concatenation | **Additive Attention Fusion** | Bahdanau-style compatibility function is more expressive than dot-product when modalities occupy different embedding subspaces |
| 2 | **Temporal Encoder** | Single LSTM | **Stacked GRU** | GRUs have fewer parameters than LSTMs (no cell state) and converge faster on medium datasets (Chung et al., 2014) |
| 3 | **Attention** | Uniform weights | **Narrative Attention with Exponential Recency Bias** | Learnable decay parameter encodes temporal locality as a soft prior |
| 4 | **Explainability** | None | **Integrated Gradients + Grad-CAM++ + Attention Visualisation** | Multi-level interpretability across temporal, spatial, and input dimensions |

---


## Key Results

| Metric | Value |
|--------|-------|
| BLEU-4 | 0.77 |
| Frame L1 Loss | 1.2801 |
| Training Epochs | 25 |

The model demonstrates stable convergence over 25 epochs, achieving consistent
frame reconstruction quality and moderate caption generation performance.

- **BLEU-4 (0.77)** indicates limited but meaningful narrative coherence, which is expected
  for multimodal sequence generation tasks with constrained training data.
- **Frame L1 Loss (1.28)** reflects effective visual prediction, showing the model can
  reconstruct plausible next frames from context.

> Note: Caption generation is inherently more challenging than frame prediction due to
> vocabulary sparsity and long-range dependencies.

---

## Architecture

```
frames (N, K, C, H, W)   ──►  ImageFeatureExtractor (ResNet-34)  ──►  (N, K, 256)
                                                                              │
captions (N, K, T)        ──►  CaptionFeatureExtractor (GRU)      ──►  (N, K, 256)
                                                                              │
                               ╔══════════════════════════════╗
                               ║  AdditiveAttentionFusion      ║  [Innovation 1]
                               ║  f(v,t) = w^T tanh(Wv + Ut)  ║
                               ╚══════════════════════════════╝
                                             │
                               ╔══════════════════════════════╗
                               ║  TemporalContextEncoder       ║  [Innovation 2]
                               ║  Stacked GRU (2 layers)       ║
                               ╚══════════════════════════════╝
                                             │
                               ╔══════════════════════════════╗
                               ║  NarrativeAttention           ║  [Innovation 3]
                               ║  Recency Bias: -λ|i-j|        ║
                               ╚══════════════════════════════╝
                                             │
                      ┌──────────────────────┴──────────────────┐
                      ▼                                           ▼
             FrameDecoder                                 CaptionDecoder
          (Upsample + Conv)                           (GRU + Scheduled Sampling)
                      │                                           │
           Predicted Frame                             Predicted Caption
            (N, 3, 256, 256)                               (N, T, vocab)
```

---

## Key Differences from Standard Approaches

| Design Choice | Standard | This Work | Reason |
|---|---|---|---|
| Image backbone | ResNet-50 | **ResNet-34** | Lighter model, less overfitting risk |
| Fusion | Concatenation | **Additive Attention** | Better cross-modal alignment |
| Sequence model | LSTM | **Stacked GRU** | Faster training, fewer parameters |
| Temporal attention | Sinusoidal PE | **Learnable recency bias** | Data-driven temporal prior |
| Image loss | MSE | **L1 Loss** | Less sensitive to outlier pixels |
| LR schedule | Cosine | **Step decay** | Predictable decay at fixed intervals |
| Context frames | 3 | **4** | More narrative context |

---

## Explainability

Three techniques implemented in `src/xai.py`:

1. **Integrated Gradients** (Sundararajan et al., 2017) — attributes predictions to
   input frames by integrating gradients along a path from a black baseline to the input.
   Saved to `outputs/xai/integrated_gradients.png`.

2. **Grad-CAM++** (Chattopadhay et al., 2018) — enhanced spatial localisation using
   second-order gradients on the ResNet-34 final conv layer.
   Saved to `outputs/xai/gradcam_pp.png`.

3. **Narrative Attention Scores** — visualises the recency-biased attention weights
   showing per-frame importance in the prediction.
   Saved to `outputs/xai/attention_scores.png`.

---

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run full notebook
jupyter notebook experiments.ipynb

# Load saved model without retraining
ckpt = torch.load('saved_models/best.pt', map_location=device)
net.load_state_dict(ckpt['net_state'])
```



---

## Configuration (see `settings.yaml`)

| Parameter | Value |
|---|---|
| Context frames K | 4 |
| Image resolution | 256×256 |
| Backbone | ResNet-34 |
| Embedding size | 256 |
| Batch size | 12 |
| Learning rate | 3e-4 |
| LR schedule | StepLR (γ=0.5, step=8) |
| Epochs | 25 |
| Optimiser | Adam |

---

## Repository Structure

```
## Repository Structure


Multimodal Neural Network/
├── README.md # Project overview, methodology, and results
├── experiments.ipynb # End-to-end pipeline: training, evaluation, XAI
├── settings.yaml # Model configuration and hyperparameters
├── requirements.txt # Python dependencies

├── src/
│ ├── architecture.py # Core model (ResNet34 + GRU + Attention + Recency Bias)
│ ├── runner.py # Training and validation routines
│ ├── xai.py # Explainability (Integrated Gradients, Grad-CAM++, Attention)
│ └── helpers.py # Utilities: preprocessing, metrics, visualisation

├── saved_models/
│ ├── best_model.pt # Best checkpoint (lowest validation loss)
│ └── final_model.pt # Final trained model with evaluation metrics

├── outputs/
│ ├── loss_curve.png # Training vs validation loss
│ ├── metrics.csv # BLEU-4 and Frame L1 scores
│ ├── metrics_bar.png # Performance comparison chart
│ ├── ablation.csv # Ablation study results
│ ├── ablation_plot.png # Ablation visualization
│ ├── caption_lengths.png # Caption length distribution
│ ├── prediction_example.png # Model-generated frame + caption
│ ├── sample_sequence.png # Input context visualization

│ └── xai/
│ ├── integrated_gradients.png # Frame-level attribution scores
│ ├── gradcam_pp.png # Spatial attention heatmap
│ ├── attention_scores.png # Temporal importance scores
│ └── attention_matrix.png # Full attention map

---

## Dataset Reference

Oliveira, D. A. P., & Matos, D. M. (2025). *StoryReasoning Dataset: Using Chain-of-Thought
for Scene Understanding and Grounded Story Generation*. arXiv:2505.10292.
https://arxiv.org/abs/2505.10292
