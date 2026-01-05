---
layout: single
title: "Locality: The License to Focus"
excerpt: "Why 3×3 kernels beat global attention, and the assumption that makes efficient ML possible"
toc: true
toc_sticky: true
math: true
---

*Subscribe to [Software Bits](https://softwarebits.substack.com/) to get new articles in your inbox.*

---

Here's a puzzle.

A transformer with full attention sees every token. A CNN with 3×3 kernels sees 9 pixels at a time. The transformer has strictly more information.

Yet for images, the CNN often wins—in accuracy, speed, and memory.

How can *less* information lead to *better* results?

The answer is a single assumption: **locality**. Nearby things are more related than distant things. When this holds, you don't need to look everywhere. You can focus.

---

## The Assumption

Locality is the principle that proximity implies relevance:

$$\text{relevance}(x, y) \propto \frac{1}{\text{distance}(x, y)}$$

In images, neighboring pixels are more correlated than distant ones. In text, recent words matter more than ancient context. In molecules, bonded atoms interact more strongly than distant ones.

This isn't always true—hence the "assumption." But when it holds, it's a license to skip computation.

**Global attention**: Look at everything. $O(n^2)$ comparisons.

**Local attention**: Look at neighbors. $O(n \cdot w)$ comparisons, where $w$ is the window size.

If $w \ll n$, that's a massive speedup. Mistral uses $w = 4096$ on sequences of 32K tokens—an **87% memory reduction** compared to full attention.

---

## Convolutions: Locality as Architecture

CNNs are locality machines.

A 3×3 kernel sees exactly 9 pixels—the immediate neighborhood. The assumption: what happens at pixel $(i, j)$ depends primarily on pixels near $(i, j)$.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   FULL ATTENTION                    3×3 CONVOLUTION             │
│                                                                 │
│   Query at (i,j) attends to:       Kernel at (i,j) sees:       │
│                                                                 │
│   ┌─────────────────────┐          ┌─────────────────────┐     │
│   │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │          │ . . . . . . . . . . │     │
│   │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │          │ . . . . . . . . . . │     │
│   │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │          │ . . . ■ ■ ■ . . . . │     │
│   │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │          │ . . . ■ ● ■ . . . . │     │
│   │ ■ ■ ■ ■ ● ■ ■ ■ ■ ■ │          │ . . . ■ ■ ■ . . . . │     │
│   │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │          │ . . . . . . . . . . │     │
│   │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │          │ . . . . . . . . . . │     │
│   └─────────────────────┘          └─────────────────────┘     │
│                                                                 │
│   100 positions attended            9 positions attended        │
│   O(n²) complexity                  O(n) complexity             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

For a 224×224 image:
- **Full attention**: 50,176 × 50,176 = **2.5 billion** attention weights per layer
- **3×3 convolution**: 9 weights per position = **450K** operations per layer

That's **5,000× less computation**—assuming locality holds.

### Why Small Kernels Work

VGG (2014) showed that stacking small kernels beats using large ones.

Two 3×3 layers have an effective receptive field of 5×5. Three 3×3 layers reach 7×7. But:

| Approach | Receptive Field | Parameters |
|----------|----------------|------------|
| One 7×7 kernel | 7×7 | 49 × C² |
| Three 3×3 kernels | 7×7 | 3 × 9 × C² = 27 × C² |

Same receptive field, **45% fewer parameters**, plus nonlinearities between layers (more expressiveness).

Locality suggests: build up context gradually from neighbors, not all at once from everywhere.

---

## Sliding Window Attention: Locality for Sequences

Transformers pay $O(n^2)$ for attention. For a 100K-token document, that's 10 billion attention computations per layer.

**Sliding window attention** applies the locality assumption to sequences:

$$\text{Attention}(q_i) = \text{softmax}\left(\frac{q_i K_{[i-w:i+w]}^T}{\sqrt{d}}\right) V_{[i-w:i+w]}$$

Each token only attends to tokens within window $w$.

### Mistral: Locality at Scale

Mistral 7B uses a window of 4,096 tokens. The complexity savings:

| Sequence Length | Full Attention | Window Attention | Memory Saved |
|-----------------|---------------|------------------|--------------|
| 8K tokens | 64M ops | 33M ops | 50% |
| 32K tokens | 1B ops | 131M ops | 87% |
| 128K tokens | 16B ops | 524M ops | 97% |

But how does Mistral handle long-range dependencies if each token only sees 4K neighbors?

**Answer: Layer stacking propagates information.**

After one layer, token $i$ has seen tokens $[i-w, i+w]$.
After two layers, it's seen $[i-2w, i+2w]$ (indirectly).
After $L$ layers, it's seen $[i-Lw, i+Lw]$.

With 32 layers and $w = 4096$, the effective receptive field is **131K tokens**—far exceeding the context length. Locality enables efficiency; depth enables global context.

### Longformer: Locality + Global Tokens

Longformer adds a twist: most attention is local, but some "global" tokens attend everywhere.

```
[ CLS ] ←────────── attends to all ──────────→
   ↓
[ token ] [ token ] [ token ] [ token ] [ token ] ...
   └──┬──┘
    local attention only (512 tokens)
```

The `[CLS]` token (used for classification) attends globally. Regular tokens attend locally.

This captures the intuition: most tokens only need local context, but some aggregation points need the full picture.

---

## Dilated Convolutions: Locality with Reach

What if you need wider context without more computation?

**Dilated (atrous) convolutions** space out the kernel:

```
Standard 3×3:           Dilated 3×3 (rate=2):
■ ■ ■                   ■ . ■ . ■
■ ● ■                   . . . . .
■ ■ ■                   ■ . ● . ■
                        . . . . .
Receptive: 3×3          ■ . ■ . ■
9 parameters            Receptive: 5×5
                        Still 9 parameters!
```

A dilation rate of 2 doubles the receptive field with zero additional parameters. Stack dilated convolutions at rates 1, 2, 4, 8... and you exponentially expand the receptive field while keeping computation linear.

This is how WaveNet generates audio: dilated convolutions capture context spanning thousands of samples without the cost of full attention.

**Locality with holes**: You still assume nearby matters more, but you sample at multiple scales to capture different neighborhood sizes.

---

## Hierarchical Processing: Coarse-to-Fine Locality

Another locality pattern: process at multiple resolutions.

**Image pyramids** downsample the input:

```
Level 0: 224×224  →  [Conv]  →  Features
         ↓ pool
Level 1: 112×112  →  [Conv]  →  Features
         ↓ pool
Level 2:  56×56   →  [Conv]  →  Features
         ↓ pool
Level 3:  28×28   →  [Conv]  →  Features (global context)
```

At each level, a 3×3 kernel sees the same relative neighborhood, but in absolute terms:
- Level 0: 3×3 = 9 pixels
- Level 1: 3×3 = 36 original pixels (2× downsampled)
- Level 2: 3×3 = 144 original pixels
- Level 3: 3×3 = 576 original pixels

Local operations at coarse scales = global context at fine scales.

This is why CNNs end with global pooling. By the final layer, "local" means "the whole image."

---

## Locality-Sensitive Hashing: Similarity as Locality

The locality principle extends beyond spatial proximity.

**Locality-Sensitive Hashing (LSH)** exploits locality in *feature space*: similar vectors should hash to the same bucket.

For nearest-neighbor search in a billion vectors:
- **Brute force**: Compare query to all billion. $O(n)$.
- **LSH**: Hash query, check its bucket. $O(1)$ expected.

The trick: design hash functions where $P(\text{same hash} | \text{similar}) > P(\text{same hash} | \text{different})$.

Random projections work: similar vectors project similarly. Bucket by projection sign, and similar items cluster.

FAISS, the standard library for vector search, uses LSH and related techniques to search billions of vectors in milliseconds.

**Locality in feature space**: Assume relevant items are similar. Hash to find them fast.

---

## Memory Hierarchy: Hardware Locality

Locality isn't just algorithmic—it's baked into hardware.

```
┌───────────────────────────────────────────────────────────────┐
│                     MEMORY HIERARCHY                          │
├───────────────────────────────────────────────────────────────┤
│   Registers     │  ~1 cycle   │  ~1 KB    │  Fastest         │
│   L1 Cache      │  ~4 cycles  │  ~32 KB   │                  │
│   L2 Cache      │  ~12 cycles │  ~256 KB  │                  │
│   L3 Cache      │  ~40 cycles │  ~8 MB    │                  │
│   RAM           │  ~200 cycles│  ~32 GB   │                  │
│   Disk/SSD      │  ~10⁵ cycles│  ~1 TB    │  Slowest         │
└───────────────────────────────────────────────────────────────┘
```

Accessing RAM is **50× slower** than L1 cache. Caches work by betting on locality: if you accessed address $x$, you'll probably access $x+1$ soon.

Neural networks that respect locality get cache benefits for free:
- Convolutions access contiguous memory (spatial locality)
- Batched operations reuse weights (temporal locality)
- Sliding window fits in fast memory (size locality)

FlashAttention's magic isn't just algorithmic—it's about fitting attention tiles in GPU SRAM instead of HBM. That's locality-aware computation.

---

## When Locality Fails

Locality is an assumption. It's often right, but not always.

**Long-range dependencies in language**: "The trophy didn't fit in the suitcase because *it* was too big." Resolving "it" requires global context.

**Non-local patterns in images**: Symmetry (a face's eyes are far apart but correlated). Textures that repeat across the image.

**Global constraints**: Physics simulations where distant particles interact. Graph problems where any node might connect to any other.

When locality fails, you have options:

1. **Accept the cost**: Use full attention where necessary
2. **Hybrid approaches**: Longformer's global tokens, sparse attention patterns
3. **Hierarchical locality**: Coarse scales capture global; fine scales stay local
4. **Learn the sparsity**: Let the model discover which long-range connections matter

The Vision Transformer (ViT) showed that full attention can beat CNNs on images—if you have enough data. Locality is a helpful prior, not a law of nature.

---

## Locality vs. Other Properties

How does locality relate to the series?

| Property | What It Says | Locality Connection |
|----------|--------------|---------------------|
| Symmetry | Position doesn't matter | Locality says *distance* matters |
| Sparsity | Most elements are zero | Locality induces sparsity (attend only to neighbors) |
| Smoothness | Gradients are bounded | Locality can aid gradient flow (local connections = shorter paths) |
| **Locality** | **Nearby > distant** | **The assumption itself** |

Locality and symmetry are complementary:
- **Translation symmetry**: The *same* operation everywhere (weight sharing)
- **Locality**: Operation only looks at *nearby* positions

CNNs have both. That's why they're so efficient: weight sharing (symmetry) + small kernels (locality).

---

## Designing with Locality

When building models, ask:

**1. Does locality hold in my domain?**

- Images: Yes (pixels are spatially correlated)
- Language: Partially (recent context matters more, but long-range exists)
- Graphs: Depends (local neighborhoods, but diameter matters)
- Time series: Usually (recent past predicts near future)
- Tabular data: No (features have no spatial structure)

**2. What's the right window size?**

- Too small: Miss important context
- Too large: Pay unnecessary computation
- Adaptive: Let the model learn (attention mechanisms)

**3. Can I use hierarchical locality?**

- Pool/downsample to create coarse representations
- Local operations at coarse scales = global at fine scales
- Cheaper than full global attention

**4. Do I need global escape hatches?**

- CLS tokens, global tokens in sparse attention
- Every-N-layer global attention
- Explicit long-range skip connections

---

## The Takeaway

Locality is the license to focus.

$$\text{If nearby matters more than distant} \implies \text{only look at nearby} \implies \text{massive speedup}$$

This assumption powers:
- **3×3 convolutions**: 5,000× less computation than full attention on images
- **Sliding window attention**: 87% memory reduction for long sequences
- **Dilated convolutions**: Exponential receptive field, linear parameters
- **LSH**: Billion-scale similarity search in milliseconds
- **Cache hierarchies**: 50× speedup from memory locality

The pattern: if distance matters, exploit it. Look nearby first. Expand only when needed.

When locality holds, you don't need to see everything. You just need to see the right neighborhood.

The algebra isn't abstract. It's why your phone processes images in real-time while full attention would take minutes.

---

*Next: [Stochasticity: The Regularizer in Disguise](stochasticity.html)*

*Previous: [Smoothness: The License to Go Deep](smoothness.html)*

---

## Further Reading

- [Simonyan & Zisserman, "Very Deep Convolutional Networks" (VGG, 2014)](https://arxiv.org/abs/1409.1556) — Why small kernels work
- [Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions" (2015)](https://arxiv.org/abs/1511.07122) — Dilated convolutions
- [Beltagy et al., "Longformer" (2020)](https://arxiv.org/abs/2004.05150) — Sliding window + global attention
- [Jiang et al., "Mistral 7B" (2023)](https://arxiv.org/abs/2310.06825) — Sliding window attention at scale
- [Van den Oord et al., "WaveNet" (2016)](https://arxiv.org/abs/1609.03499) — Dilated convolutions for audio
- [Dao et al., "FlashAttention" (2022)](https://arxiv.org/abs/2205.14135) — Memory-aware attention via tiling
- [Indyk & Motwani, "Approximate Nearest Neighbors" (1998)](https://dl.acm.org/doi/10.1145/276698.276876) — Locality-sensitive hashing
