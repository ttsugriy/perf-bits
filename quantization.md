---
layout: single
title: "Quantization: The License to Approximate"
excerpt: "Why 4 bits can do the work of 32, and the mathematical reason neural networks tolerate imprecision"
toc: true
toc_sticky: true
math: true
---

*Subscribe to [Software Bits](https://softwarebits.substack.com/) to get new articles in your inbox.*

---

Here's a fact that shouldn't be true.

Take a 70-billion-parameter language model trained in FP16. Reduce every weight to 4 bits—a 4× compression. The model still works. Not "barely works." Works *nearly as well*.

Llama 2 70B in FP16: 140 GB. In 4-bit: 35 GB. Fits on a consumer GPU. Quality loss: ~1-2% on most benchmarks.

We threw away 75% of the precision. Where did it go? Why didn't the model collapse?

The answer is **quantization**—the principle that neural networks are over-specified. They don't need 32 bits of precision because the function they compute is robust to small perturbations.

Quantization is the license to approximate. And it's why you can run GPT-class models on your laptop.

---

## The Property

**Quantization** is the mapping of continuous (or high-precision) values to a discrete set of levels:

$$Q(x) = \Delta \cdot \text{round}\left(\frac{x}{\Delta}\right)$$

where $\Delta$ is the step size (quantization interval).

A 32-bit float can represent ~4 billion distinct values. An 8-bit integer: 256. A 4-bit integer: 16.

The naive view: fewer values = less precision = worse accuracy.

The truth: neural networks are massively over-parameterized. The weights don't need to be precise—they need to be *approximately right*.

$$\text{Over-parameterization} \implies \text{Redundancy} \implies \text{Quantization tolerance}$$

---

## Why It Works: The Robustness of Neural Networks

Three properties explain why quantization works.

### 1. Training Noise Exceeds Quantization Noise

Consider training with SGD. Each gradient is noisy—computed from a small batch, not the full dataset. The weight updates have variance proportional to learning rate and batch size.

If training succeeds despite this noise, the final solution must be robust to small perturbations. Quantization just adds more small perturbations—of the same magnitude as what training already handled.

**Gradient noise during training**: variance $\sigma^2 \propto \frac{\eta}{B}$

**Quantization noise**: variance $\sigma^2 \propto \frac{\Delta^2}{12}$

If $\Delta$ is small enough that quantization noise is comparable to training noise, accuracy is preserved.

### 2. Flat Minima Are Robust

Modern neural networks converge to flat minima—regions where the loss changes slowly with parameters (see [Smoothness](smoothness.html)).

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   QUANTIZATION IN FLAT VS SHARP MINIMA                         │
│                                                                 │
│        Loss                                                     │
│          ↑                                                      │
│          │     ╱╲                                               │
│          │    ╱  ╲         ╱╲                                   │
│          │   ╱    ╲  ┌────╱  ╲────┐                             │
│          │  ╱      ╲ │    flat    │                             │
│          │ ╱        ╲│   minimum  │                             │
│          │╱          ●....●....●  │   ← Quantized weights      │
│          │    sharp      ^            stay in low-loss region  │
│          │   minimum  original                                  │
│          │   ↑                                                  │
│          │   Quantization causes large loss change             │
│          └──────────────────────────→ Parameters               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Flat minima = robustness to perturbation = quantization tolerance.

### 3. Redundancy in Over-Parameterized Networks

A network with 70 billion parameters doesn't need 70 billion degrees of freedom to represent its function. The effective dimensionality is much lower.

Evidence:
- Neural networks can be pruned by 90% with minimal accuracy loss
- Low-rank approximations capture most of the information (see [Separability](separability.html))
- Different random initializations converge to equivalent solutions

This redundancy means quantization can destroy some information without destroying the function.

---

## The Precision Hierarchy

Not all bits are equal.

| Format | Bits | Range | Precision | Use Case |
|--------|------|-------|-----------|----------|
| FP32 | 32 | $\pm 3.4 \times 10^{38}$ | ~7 decimal digits | Training (legacy) |
| BF16 | 16 | $\pm 3.4 \times 10^{38}$ | ~3 decimal digits | Training (modern) |
| FP16 | 16 | $\pm 65504$ | ~4 decimal digits | Mixed precision |
| INT8 | 8 | -128 to 127 | 256 levels | Inference |
| INT4 | 4 | -8 to 7 | 16 levels | Compressed inference |

The trend: training uses 16+ bits (for gradient precision), inference uses 8 or fewer (for speed).

### Why BF16 for Training?

BF16 keeps FP32's exponent range (8 bits) but truncates the mantissa (7 bits instead of 23).

$$\text{BF16} = \text{sign} + 8\text{-bit exponent} + 7\text{-bit mantissa}$$

Range matters more than precision for training. Gradients can span many orders of magnitude; representing them all matters more than representing each precisely.

FP16 has better precision but smaller range—gradients can overflow. BF16 trades precision for range, matching training's needs.

### Why INT8 for Inference?

At inference, we're not computing gradients. We're just multiplying weights by activations and summing.

Integer arithmetic is:
- **Faster**: INT8 multiply-add is ~2× faster than FP16
- **Smaller**: 4× less memory than FP32, 2× less than FP16
- **More power-efficient**: Simpler circuits consume less energy

The cost: only 256 distinct values instead of billions.

But 256 values, chosen well, are enough.

---

## Quantization Techniques

How do we map continuous weights to discrete levels?

### Uniform Quantization

The simplest approach: divide the range into equal intervals.

$$Q(x) = \text{clamp}\left(\text{round}\left(\frac{x - z}{s}\right), 0, 2^b - 1\right)$$

where:
- $s$ = scale factor (step size)
- $z$ = zero point
- $b$ = number of bits

**Symmetric quantization**: zero point = 0, range is $[-\alpha, \alpha]$
**Asymmetric quantization**: zero point $\neq 0$, range is $[\beta_{min}, \beta_{max}]$

Symmetric is simpler; asymmetric handles skewed distributions better.

### Per-Tensor vs. Per-Channel

**Per-tensor**: One scale for the entire weight matrix. Simple but loses precision if ranges vary across channels.

**Per-channel**: Different scale per output channel. More parameters but much better accuracy.

```
Per-tensor:  W_quant = round(W / s)           # one s for all
Per-channel: W_quant[i] = round(W[i] / s[i])  # different s per row
```

Per-channel is standard for weight quantization. The overhead of storing one scale per channel is negligible.

### Post-Training Quantization (PTQ)

Quantize a trained FP32 model without retraining:

1. Collect calibration data (subset of training data)
2. Run forward passes to collect activation statistics
3. Choose scales that minimize quantization error
4. Convert weights and activations to integers

**Pros**: Fast, no retraining needed
**Cons**: Accuracy loss for aggressive quantization (INT4)

### Quantization-Aware Training (QAT)

Simulate quantization during training:

```python
def quantize_aware_forward(x, w, s):
    # Fake quantization: quantize then dequantize
    w_quant = round(w / s) * s  # simulates integer arithmetic
    return x @ w_quant
```

The backward pass uses straight-through estimators—gradients flow through the rounding operation as if it were identity.

**Pros**: Better accuracy, especially for INT4
**Cons**: Requires retraining, more complex pipeline

---

## Advanced: LLM Quantization

Large language models have unique quantization challenges.

### The Outlier Problem

LLM activations contain **outliers**—values 10-100× larger than typical values. These appear in specific channels and persist across inputs.

Uniform quantization fails: either clip the outliers (catastrophic accuracy loss) or expand the range (waste precision on the common case).

### GPTQ: Optimal Weight Quantization

GPTQ quantizes weights one at a time, correcting for quantization error:

$$w_{quantized} = \arg\min_q \|W - q\|_H$$

where $H$ is the Hessian (approximated from calibration data).

Key insight: when you quantize one weight, adjust the remaining unquantized weights to compensate. This reduces cumulative error.

Result: 3-4 bit quantization with minimal accuracy loss. Llama 2 70B runs in ~35 GB.

### AWQ: Activation-Aware Quantization

Not all weights matter equally. Weights connected to outlier channels need more precision.

AWQ identifies "salient" weights (those that would cause large activation errors if quantized poorly) and scales them before quantization:

$$w' = w \cdot s, \quad x' = x / s$$

The multiplication is mathematically equivalent, but the scaled weights have a better distribution for quantization.

Result: 4-bit quantization with ~0.5% accuracy loss.

### KV Cache Quantization

At inference, the key-value cache grows with sequence length. For a 100K-token context:

$$\text{KV cache size} = 2 \times L \times n_{heads} \times d_{head} \times \text{seq\_len} \times \text{bytes}$$

For Llama 70B with 100K tokens: ~40 GB in FP16, ~10 GB in INT4.

Quantizing the KV cache (not just weights) is essential for long contexts.

---

## The Hardware Connection

Quantization isn't just an algorithm—it's about hardware.

### Memory Bandwidth is the Bottleneck

Modern GPUs have massive compute capacity (hundreds of TFLOPS) but limited memory bandwidth (~2 TB/s for H100).

For LLM inference:
- Each token requires reading all weights from memory
- The model computes faster than memory can feed it
- **Quantization reduces memory traffic**, directly improving throughput

4-bit weights = 4× less memory to read = ~4× faster inference (in memory-bound regimes).

### Integer Tensor Cores

Modern GPUs have specialized integer matrix multiply units:

| GPU | FP16 TFLOPS | INT8 TOPS |
|-----|-------------|-----------|
| A100 | 312 | 624 |
| H100 | 989 | 1979 |

INT8 is 2× faster than FP16. This isn't about algorithm—it's about silicon.

n-bit multiplication requires $O(n^2)$ circuit area. INT8 uses 25% of the area of FP32 multiply. The savings compound.

---

## Quantization vs. Other Efficiency Techniques

How does quantization relate to the series?

| Property | What It Removes | Quantization Connection |
|----------|----------------|-------------------------|
| Sparsity | Zero values | Quantization + pruning are complementary |
| Separability | Full-rank structure | Low-rank quantization (LoRA + quantization) |
| Smoothness | Sharp minima | Flat minima enable quantization tolerance |
| Locality | Distant dependencies | N/A |
| **Quantization** | **Precision** | **The core technique** |

Quantization removes precision. Sparsity removes values. Separability removes rank. All exploit redundancy.

### QLoRA: Combining Them All

QLoRA combines quantization with low-rank adaptation:

1. Quantize base model to 4-bit (quantization)
2. Add low-rank adapters in FP16 (separability)
3. Train only the adapters (sparsity in updates)

Result: Fine-tune a 65B model on a single 48GB GPU.

The techniques compose because they exploit different redundancies.

---

## Limits of Quantization

Quantization isn't free.

### Accuracy Degradation

| Precision | Typical Accuracy Loss |
|-----------|-----------------------|
| INT8 | < 0.5% |
| INT4 | 1-3% |
| INT3 | 3-5% |
| INT2 | 5-10%+ |

Below 4 bits, accuracy degrades rapidly. The minimum precision depends on model size—larger models tolerate more aggressive quantization.

### Task Sensitivity

Some capabilities degrade faster:
- **Math reasoning**: Sensitive to precision loss
- **Factual recall**: Moderate sensitivity
- **General language**: Robust

The degradation isn't uniform. Evaluate on your actual use case.

### Training in Low Precision

Training in INT8 is hard. Gradients need precision to accumulate small updates without rounding to zero.

$$\text{If } \Delta w < \text{quantization step} \implies w_{new} = w_{old}$$

Training typically uses BF16/FP16 minimum. Inference uses INT8/INT4.

---

## Designing with Quantization

When deploying models, ask:

**1. What precision does my task need?**
- Production LLM inference: INT4-INT8
- Real-time computer vision: INT8
- Training: BF16/FP16 minimum

**2. Which quantization technique?**
- Simple model, lots of data: PTQ
- Aggressive quantization: QAT
- LLM deployment: GPTQ/AWQ

**3. What's my bottleneck?**
- Memory bandwidth: Quantization helps most
- Compute: INT8 tensor cores help
- Neither: Quantization has diminishing returns

**4. Can I combine with other techniques?**
- Quantization + pruning
- Quantization + low-rank (QLoRA)
- Quantization + distillation

---

## The Takeaway

Quantization is the license to approximate.

$$\text{Over-parameterization} \implies \text{Redundancy} \implies \text{Quantization tolerance}$$

This explains why:
- 4-bit LLMs work nearly as well as 16-bit (~75% compression, ~1-2% accuracy loss)
- INT8 inference is standard (2× faster, 4× smaller than FP32)
- Training noise is a feature (creates robustness to quantization)
- Flat minima matter (perturbation-robust = quantization-tolerant)

The key insight: neural networks don't need precise weights. They need *approximately correct* weights. Quantization exploits the gap between full precision and necessary precision.

When you quantize a model, you're not destroying information—you're removing redundancy that was never needed.

The 32-bit float was overkill. Four bits are enough. That's why Llama runs on your laptop.

---

*Next: Coming soon*

*Previous: [Compositionality: The Power of Depth](compositionality.html)*

---

## Further Reading

- [Dettmers et al., "LLM.int8()" (2022)](https://arxiv.org/abs/2208.07339) — Outlier-aware INT8 quantization
- [Frantar et al., "GPTQ" (2023)](https://arxiv.org/abs/2210.17323) — Optimal weight quantization for LLMs
- [Lin et al., "AWQ" (2023)](https://arxiv.org/abs/2306.00978) — Activation-aware weight quantization
- [Dettmers et al., "QLoRA" (2023)](https://arxiv.org/abs/2305.14314) — Quantized low-rank adaptation
- [Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)](https://arxiv.org/abs/1712.05877) — Foundational quantization-aware training
- [Nagel et al., "A White Paper on Neural Network Quantization" (2021)](https://arxiv.org/abs/2106.08295) — Comprehensive quantization survey
