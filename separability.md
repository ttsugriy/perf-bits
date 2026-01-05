---
layout: single
title: "Separability: The Art of Factorization"
excerpt: "Why MobileNet is 12x faster than ResNet, and how LoRA fine-tunes GPT-3 with 10,000x fewer parameters"
toc: true
toc_sticky: true
math: true
---

*Subscribe to [Software Bits](https://softwarebits.substack.com/) to get new articles in your inbox.*

---

Here's a fact that should surprise you:

MobileNetV2 uses **0.3 GFLOPs**. ResNet-50 uses **3.8 GFLOPs**. Both achieve similar accuracy on ImageNet.

That's 12x less compute for the same task.

The trick? **Separable convolutions**—factoring one expensive operation into two cheap ones.

This pattern appears everywhere. LoRA fine-tunes GPT-3 with 10,000x fewer trainable parameters. Matrix factorization powers recommendation systems. Low-rank approximations compress entire neural networks.

The principle is always the same: **if you can factor a computation, you can make it cheaper.**

---

## The Property

A matrix is separable if it can be written as a product of smaller matrices:

$$W = UV$$

where $W \in \mathbb{R}^{m \times n}$, $U \in \mathbb{R}^{m \times r}$, and $V \in \mathbb{R}^{r \times n}$.

The key insight: if $r \ll \min(m, n)$, then storing and computing with $U$ and $V$ is much cheaper than with $W$ directly.

| Representation | Parameters | Matrix-vector multiply |
|---------------|------------|----------------------|
| Full $W$ | $mn$ | $O(mn)$ |
| Factored $UV$ | $mr + rn$ | $O(mr + rn)$ |

For a 1000×1000 matrix with rank 10:
- Full: 1,000,000 parameters
- Factored: 20,000 parameters (50x smaller)

**Separability is the license to factor.**

---

## Depthwise Separable Convolutions

The canonical example in deep learning.

### Standard Convolution

A standard convolution with:
- $D_K \times D_K$ kernel
- $M$ input channels
- $N$ output channels
- $D_F \times D_F$ output feature map

Computational cost:

$$D_K^2 \cdot M \cdot N \cdot D_F^2$$

Each output pixel requires a $D_K \times D_K \times M$ dot product. There are $N \cdot D_F^2$ output values.

### Depthwise Separable Convolution

Factor the convolution into two steps:

**Step 1: Depthwise convolution**
Apply a separate $D_K \times D_K$ filter to each input channel.

Cost: $D_K^2 \cdot M \cdot D_F^2$

**Step 2: Pointwise convolution**
Apply $1 \times 1$ convolutions to combine channels.

Cost: $M \cdot N \cdot D_F^2$

Total cost:

$$D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2 = M \cdot D_F^2 \cdot (D_K^2 + N)$$

### The Reduction

The ratio of separable to standard:

$$\frac{M \cdot D_F^2 \cdot (D_K^2 + N)}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

For a $3 \times 3$ kernel with 256 output channels:

$$\frac{1}{256} + \frac{1}{9} \approx 0.115$$

That's **8-9x fewer operations**.

### In Practice

| Model | Parameters | GFLOPs | ImageNet Top-1 |
|-------|-----------|--------|----------------|
| ResNet-50 | 25.6M | 3.8 | 76.2% |
| MobileNetV2 | 3.4M | 0.3 | 71.8% |
| EfficientNet-B0 | 5.3M | 0.39 | 77.1% |

MobileNetV2 achieves 94% of ResNet-50's accuracy with 7.5x fewer parameters and 12x fewer FLOPs.

This is the power of separability: same expressiveness, fraction of the cost.

---

## Low-Rank Adaptation (LoRA)

The same principle applied to fine-tuning.

### The Problem

Fine-tuning GPT-3 means updating 175 billion parameters. That requires:
- Storing 175B gradients
- Storing optimizer states (350B+ for Adam)
- Multiple copies for different tasks

Infeasible for most practitioners.

### The Insight

Weight updates during fine-tuning are low-rank.

When you fine-tune a pretrained model, the change $\Delta W$ to each weight matrix tends to have much lower rank than the matrix itself. The model has already learned general structure; fine-tuning just adjusts it for the specific task.

### The Solution

Instead of updating the full weight matrix, add a low-rank adapter:

$$W' = W + BA$$

where:
- $W \in \mathbb{R}^{d \times k}$ is frozen (pretrained weights)
- $B \in \mathbb{R}^{d \times r}$ is trainable
- $A \in \mathbb{R}^{r \times k}$ is trainable
- $r \ll \min(d, k)$ is the rank (typically 4, 8, 16, or 32)

During training, only $B$ and $A$ are updated. $W$ stays fixed.

### The Numbers

For a weight matrix of size $d \times k$:

| Approach | Trainable Parameters |
|----------|---------------------|
| Full fine-tuning | $dk$ |
| LoRA (rank $r$) | $r(d + k)$ |

For $d = k = 4096$ and $r = 8$:
- Full: 16,777,216 parameters
- LoRA: 65,536 parameters (256x reduction)

Applied across GPT-3:

> LoRA can reduce the number of trainable parameters by **10,000 times** and the GPU memory requirement by **3 times**.
> — Hu et al., 2021

After training, merge the adapter back:

$$W_{merged} = W + BA$$

**No inference overhead.** The adapter disappears into the base weights.

---

## Why Low-Rank Works

This seems too good to be true. How can you throw away 99.99% of parameters and match full fine-tuning?

### The Intrinsic Dimension Hypothesis

Research on intrinsic dimensionality shows that neural network training happens in a surprisingly low-dimensional subspace.

The intuition: pretrained models have already learned the hard part. Fine-tuning is just steering—you don't need to change the entire weight space, just find the right direction within it.

The larger and better-pretrained the model, the lower the intrinsic dimension of fine-tuning. This is why LoRA works better on larger models.

### The Manifold Hypothesis

Real data lies on low-dimensional manifolds embedded in high-dimensional space.

A 1024×1024 image has over a million pixels, but the space of "natural images" is much smaller. Neural networks learn to exploit this structure. Their weights inherit the low-rank structure of the data they model.

---

## SVD: The Optimal Factorization

How do you find the best low-rank approximation?

The Singular Value Decomposition (SVD) gives the answer:

$$W = U \Sigma V^T$$

where:
- $U$ and $V$ are orthogonal matrices
- $\Sigma$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r$

The **Eckart-Young theorem** says: the best rank-$k$ approximation (in Frobenius norm) is:

$$W_k = U_k \Sigma_k V_k^T$$

Just keep the top $k$ singular values and their corresponding vectors.

### Compression via SVD

This enables direct model compression:

1. Train a full network
2. Apply SVD to each weight matrix
3. Keep only the top-$k$ singular components
4. (Optional) Fine-tune to recover accuracy

The singular value spectrum tells you how compressible a matrix is. If singular values decay rapidly, most of the "information" is in the top components.

---

## Tensor Decomposition

Weights aren't always matrices. Convolution kernels are 4D tensors: (output channels, input channels, height, width).

Tensor decompositions generalize matrix factorization:

### CP Decomposition

Write a tensor as a sum of rank-1 tensors:

$$\mathcal{T} \approx \sum_{r=1}^{R} a_r \otimes b_r \otimes c_r \otimes d_r$$

Each rank-1 component is an outer product of vectors.

### Tucker Decomposition

Factor into a small core tensor multiplied by matrices along each mode:

$$\mathcal{T} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C \times_4 D$$

Both enable compression of convolutional layers beyond what matrix SVD can achieve.

---

## Matrix Factorization for Recommendations

The Netflix Prize (2006-2009) popularized matrix factorization for recommendations.

### The Setup

You have a sparse matrix of user ratings:

$$R_{ij} = \text{rating by user } i \text{ for movie } j$$

Most entries are missing. You want to predict them.

### The Model

Assume low-rank structure:

$$R \approx UV^T$$

where:
- $U \in \mathbb{R}^{\text{users} \times k}$ — user embeddings
- $V \in \mathbb{R}^{\text{movies} \times k}$ — movie embeddings
- $k$ is small (50-200)

Each user is a point in $k$-dimensional space. Each movie is a point in the same space. The rating is their dot product.

### Why It Works

The low-rank assumption captures **latent factors**: action vs. drama, cerebral vs. escapist, etc.

Users with similar taste have similar embeddings. Movies with similar audiences have similar embeddings. The factorization discovers this structure automatically.

**This is the same insight as word embeddings.** Words that appear in similar contexts get similar vectors. "king" − "man" + "woman" ≈ "queen" works because the embedding captures separable semantic dimensions.

---

## The Linear Attention Trade-off

Separability isn't always free. Sometimes it costs expressiveness.

### Standard Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

The $QK^T$ term is $O(n^2)$ for sequence length $n$.

### Linear Attention

What if we could factor this?

Replace softmax with a kernel:

$$\text{Attention}(Q, K, V) \approx \phi(Q)(\phi(K)^T V)$$

By computing $\phi(K)^T V$ first (which is $d \times d$), you avoid the $n \times n$ attention matrix.

Complexity drops from $O(n^2 d)$ to $O(nd^2)$.

### The Low-Rank Dilemma

But there's a catch: **the factored representation has lower rank**.

The feature map $\phi(K)^T V$ is a $d \times d$ matrix—at most rank $d$. Standard attention can represent any $n \times n$ attention pattern.

This limits what patterns linear attention can express. Research on Rank-Augmented Linear Attention (RALA) attempts to close this gap, achieving competitive accuracy while maintaining efficiency.

**Separability trades rank for speed.** When the true computation is inherently low-rank, you win. When it isn't, you sacrifice expressiveness.

---

## When Separability Works

The pattern emerges:

| Domain | What's Factored | Why It Works |
|--------|----------------|--------------|
| Depthwise separable | Spatial × channel | Spatial and channel patterns are ~independent |
| LoRA | Weight update | Fine-tuning is low-rank steering |
| Matrix factorization | User × item | Preferences have latent structure |
| SVD compression | Weight matrices | Overparameterized networks are low-rank |

Separability works when:

1. **Structure is approximately factored**: Spatial patterns don't depend strongly on channel identity. User preferences decompose into latent factors.

2. **Overparameterization exists**: Neural networks have more parameters than necessary. The excess can be factored out.

3. **The task is refinement, not revolution**: Fine-tuning adjusts pretrained weights; it doesn't reinvent them.

---

## The Takeaway

Separability is the art of factorization.

$$W = UV \quad \text{where } r \ll \min(m, n)$$

When you can write an expensive operation as a product of cheaper ones, you win.

- **Depthwise separable convolutions**: Factor spatial and channel mixing → 8-9x fewer operations
- **LoRA**: Factor weight updates as low-rank → 10,000x fewer trainable parameters
- **Matrix factorization**: Factor user-item matrices → discover latent structure
- **SVD compression**: Factor weight matrices → remove redundancy

The underlying insight: **most structure in the world is low-rank**.

High-dimensional data lives on low-dimensional manifolds. Overparameterized networks learn low-rank weight matrices. Fine-tuning happens in low-dimensional subspaces.

Separability exploits this. Instead of computing with the full representation, factor it and compute with the factors.

The algebra isn't abstract. It's why your phone can run neural networks.

---

*Previous article: [Sparsity: The License to Skip](sparsity.html)*

---

## Further Reading

- [Howard et al., "MobileNets" (2017)](https://arxiv.org/abs/1704.04861) — Depthwise separable convolutions
- [Hu et al., "LoRA: Low-Rank Adaptation" (2021)](https://arxiv.org/abs/2106.09685) — Fine-tuning with low-rank adapters
- [Koren et al., "Matrix Factorization Techniques" (2009)](https://ieeexplore.ieee.org/document/5197422) — Netflix Prize and recommendations
- [Kolda & Bader, "Tensor Decompositions" (2009)](https://epubs.siam.org/doi/10.1137/07070111X) — CP and Tucker decomposition
- [Aghajanyan et al., "Intrinsic Dimensionality" (2021)](https://arxiv.org/abs/2012.13255) — Why fine-tuning is low-dimensional
