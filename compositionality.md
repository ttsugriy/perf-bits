---
layout: single
title: "Compositionality: The Power of Depth"
excerpt: "Why stacking layers creates exponential expressiveness"
toc: true
toc_sticky: true
math: true
---

*Subscribe to [Software Bits](https://softwarebits.substack.com/) to get new articles in your inbox.*

---

Here's a result that explains why deep learning works.

There exist functions that a deep ReLU network with $k^3$ layers and only $O(1)$ neurons per layer can compute, but which require a shallow network with $O(k)$ layers to have at least $\Omega(2^k)$ neurons.

For $k = 30$: the deep network needs ~90 neurons. A shallow network with 30 layers needs over **one billion**.

That's not a percentage improvement. It's an *exponential* gap.

The difference? **Compositionality**—the principle that complex functions are built by composing simpler ones.

---

## The Property

Compositionality means building complex things from simple pieces:

$$f = f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1$$

A deep network is a composition of layers. Each layer transforms its input; the output becomes the next layer's input.

This seems like an implementation detail. It's not. It's the source of deep learning's power.

**Shallow networks** (one hidden layer) are universal approximators—they can represent any continuous function given enough neurons. But "enough" can mean *exponentially many*.

**Deep networks** represent the same functions with polynomially many neurons. The depth creates leverage.

---

## The Exponential Gap

Why does depth help so dramatically?

### Linear Regions

A ReLU network divides input space into **linear regions**—pieces where the function is linear. More regions = more expressive.

**Shallow network** (width $w$, depth 1):
- Maximum $O(w^d)$ regions, where $d$ is input dimension

**Deep network** (width $w$, depth $L$):
- Maximum $O(w^{Ld})$ regions

The number of regions grows **exponentially with depth**. A 10-layer network can have $w^{10}$ more regions than a 1-layer network with the same width.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   SHALLOW (1 hidden layer)         DEEP (4 hidden layers)      │
│                                                                 │
│   ┌─────────────────┐              ┌─────────────────┐         │
│   │    ╱│╲          │              │ ╱╲╱╲╱╲╱╲╱╲╱╲   │         │
│   │   ╱ │ ╲         │              │╱╲╱╲╱╲╱╲╱╲╱╲╱╲  │         │
│   │  ╱  │  ╲        │              │╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲ │         │
│   │ ╱   │   ╲       │              │╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱ │         │
│   └─────────────────┘              └─────────────────┘         │
│                                                                 │
│   ~4 linear regions                ~64 linear regions          │
│   Same total neurons!                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Each layer multiplies expressiveness. Shallow networks add; deep networks multiply.

### The Formal Result

Research has proven strict separation theorems:

> There exist functions computable by networks with $\Theta(k^3)$ layers and $O(1)$ nodes per layer that cannot be approximated by networks with $O(k)$ layers unless they have $\Omega(2^k)$ nodes. — Telgarsky (2016)

This is **exponential** separation. Depth isn't just convenient—it's fundamentally more efficient for certain functions. Every additional layer of depth can halve the width required, or equivalently, double the expressiveness.

---

## Hierarchical Features

Compositionality explains why CNNs learn what they learn.

### The Feature Hierarchy

Visualizing CNN activations reveals a striking pattern:

| Layer | What It Detects | Receptive Field |
|-------|-----------------|-----------------|
| Layer 1 | Edges, colors | 3×3 pixels |
| Layer 2 | Textures, corners | ~10×10 pixels |
| Layer 3 | Parts (eyes, wheels) | ~50×50 pixels |
| Layer 4 | Objects (faces, cars) | ~100×100 pixels |
| Layer 5 | Scenes, contexts | Whole image |

Each layer builds on the previous. Edges compose into textures. Textures compose into parts. Parts compose into objects.

This isn't designed—it *emerges* from composition. The network discovers that hierarchical features are efficient for the task.

### Why Hierarchy is Efficient

Consider detecting a face:

**Without composition**: Learn all possible face pixel patterns directly. Exponentially many patterns (lighting, pose, identity, expression...).

**With composition**:
1. Learn edge detectors (few patterns)
2. Compose edges into texture detectors (reuse edges)
3. Compose textures into part detectors (reuse textures)
4. Compose parts into face detectors (reuse parts)

Each level reuses the previous level. A nose detector doesn't need to re-learn edges—it builds on existing edge detectors.

This is why deep networks need fewer parameters than shallow ones: **feature reuse through composition**.

---

## Mathematical Intuition

### Exponentiation Analogy

Consider computing $x^{1024}$.

**Direct approach**: Multiply $x \times x \times \cdots \times x$ (1,023 multiplications).

**Compositional approach** (repeated squaring):
- $x^2$, then $(x^2)^2 = x^4$, then $(x^4)^2 = x^8$, ...
- 10 squarings to reach $x^{1024}$

The compositional approach is $O(\log n)$; the direct approach is $O(n)$. For $n = 1024$, that's 10 operations versus 1,023.

This is exactly what depth provides: **reusing intermediate computations**. The direct approach treats each multiplication independently. The compositional approach builds each result from the previous.

Deep networks do the same thing: early layers compute features that later layers reuse, rather than recomputing everything from scratch.

### Circuit Complexity

Computer science has long studied this through circuits.

**AND-OR circuits** (like neural networks with thresholds) have known depth-efficiency results:

- Some functions require exponentially many gates at depth 2
- The same functions need only polynomially many gates at depth $O(\log n)$

The parity function (XOR of $n$ bits) is a classic example: it's trivial with logarithmic depth but requires exponentially many gates at constant depth.

Neural networks inherit these complexity advantages.

---

## Composition in Modern Architectures

Every successful deep learning architecture is built on composition.

### Transformers

A transformer is a composition of identical blocks:

```python
def transformer(x, num_layers):
    for _ in range(num_layers):
        x = x + attention(layer_norm(x))
        x = x + ffn(layer_norm(x))
    return x
```

Each block composes attention (global mixing) with FFN (local processing). Stacking blocks compounds their effects.

GPT-3 uses 96 layers. Each layer builds on previous representations. The first layers might capture syntax; later layers capture semantics, then reasoning patterns.

### ResNets

ResNets explicitly frame composition as residual learning:

$$h_{l+1} = h_l + f_l(h_l)$$

Each layer adds a residual. The final output is the sum of all residuals—a composition of refinements.

The residual framing makes optimization easier (see [Smoothness](smoothness.html)), but it's still composition that provides expressiveness.

### U-Nets

U-Nets compose encoder and decoder:

```
Input → Encode → Encode → Encode → Bottleneck
                                      ↓
Output ← Decode ← Decode ← Decode ←───┘
         (+skip)  (+skip)  (+skip)
```

The encoder composes down to abstract representations. The decoder composes back up to detailed outputs. Skip connections allow high-frequency details to bypass the bottleneck.

This compositional structure is why U-Nets excel at dense prediction (segmentation, super-resolution).

---

## The Efficiency of Modularity

Composition enables **modularity**—reusable components.

### Transfer Learning

A pretrained CNN has learned hierarchical features. The early layers (edges, textures) transfer to almost any vision task. Only the later layers need task-specific training.

This works because of composition: early layers are *general* (useful across tasks), late layers are *specific* (task-dependent). You can compose new heads onto pretrained bodies.

Without compositional structure, every task would require learning from scratch.

### Mixture of Experts

MoE (see [Sparsity](sparsity.html)) exploits composition differently:

$$y = \sum_i g_i(x) \cdot E_i(x)$$

Different experts handle different inputs. The composition is *conditional*—different subnetworks activate for different examples.

This is composition with routing: the network composes specialized modules dynamically.

---

## Why Shallow Networks Fail

If shallow networks are universal approximators, why not just use them?

### The Memorization Problem

A shallow network with enough neurons can memorize any training set. But memorization requires one feature per training example—no generalization.

Deep networks can't memorize as easily. Their compositional structure forces them to find patterns that apply across examples.

Paradoxically, **limited capacity forces generalization**.

### The Feature Reuse Problem

Shallow networks can't reuse features. Every output unit connects directly to inputs—there's no intermediate representation to share.

Consider detecting 1000 object classes:
- **Shallow**: 1000 independent classifiers, each learning edges from scratch
- **Deep**: One shared edge detector, reused by all classifiers

The deep network amortizes feature learning across classes. The shallow network does redundant work.

---

## Depth-Width Tradeoffs

Depth and width aren't interchangeable.

### What Width Provides

- **Parallelism**: More neurons per layer = more parallel computation
- **Feature diversity**: More features detected simultaneously
- **Optimization ease**: Wider networks have smoother loss landscapes

### What Depth Provides

- **Abstraction**: More layers = more levels of abstraction
- **Efficiency**: Exponentially more expressiveness per parameter
- **Composition**: Complex functions from simple pieces

### The Modern Consensus

State-of-the-art architectures balance both:

| Model | Depth | Width | Key Insight |
|-------|-------|-------|-------------|
| ResNet-152 | 152 layers | 64-2048 | Deep with residuals |
| GPT-3 | 96 layers | 12288 | Very wide + very deep |
| ViT-Large | 24 layers | 1024 | Moderate both |
| Mamba | 48 layers | 2560 | Linear attention + depth |

The trend: both depth and width matter, but depth provides unique efficiency gains.

---

## Designing with Compositionality

When building architectures, ask:

**1. What's the natural hierarchy in my problem?**

- Vision: pixels → edges → textures → parts → objects
- Language: characters → words → phrases → sentences → paragraphs
- Audio: samples → frames → phonemes → words → speech

Design depth to match this hierarchy.

**2. Can I reuse components?**

- Standard blocks (ResNet block, Transformer block)
- Pretrained components (frozen encoders, adapters)
- Conditional routing (MoE, dynamic networks)

Composition + reuse = efficiency.

**3. Where should abstraction happen?**

- Pooling layers reduce spatial dimensions
- Attention layers mix global information
- Bottleneck layers compress representations

Strategic composition points control information flow.

---

## The Takeaway

Compositionality is the power of depth.

$$f = f_n \circ f_{n-1} \circ \cdots \circ f_1$$

This simple principle explains:
- Why deep networks need exponentially fewer parameters than shallow ones
- Why CNNs learn edges → textures → parts → objects
- Why Transformers stack identical blocks
- Why transfer learning works
- Why modularity is efficient

The key insight: each layer multiplies expressiveness. Depth doesn't add power—it *compounds* it.

A 100-layer network isn't 100× more powerful than a 1-layer network. It's $2^{100}$ more powerful (in terms of representable linear regions).

This is why deep learning is "deep." Not because deep sounds impressive, but because composition is exponentially more efficient than enumeration.

When you need to represent complex functions, don't make the network wider. Make it deeper. Let composition do the heavy lifting.

---

*Next: [Quantization: The License to Approximate](quantization.html)*

*Previous: [Stochasticity: The Regularizer in Disguise](stochasticity.html)*

---

## Further Reading

- [Montúfar et al., "On the Number of Linear Regions of Deep Neural Networks" (2014)](https://arxiv.org/abs/1402.1869) — Why depth creates exponentially more linear regions
- [Telgarsky, "Benefits of Depth in Neural Networks" (2016)](https://arxiv.org/abs/1602.04485) — Formal separation between deep and shallow
- [Mhaskar & Poggio, "Deep vs. Shallow Networks" (2016)](https://arxiv.org/abs/1608.03287) — Approximation theory perspective
- [Zeiler & Fergus, "Visualizing and Understanding CNNs" (2014)](https://arxiv.org/abs/1311.2901) — Hierarchical feature visualization
- [Arora et al., "Understanding Deep Neural Networks with Rectified Linear Units" (2018)](https://arxiv.org/abs/1611.01491) — Zonotope theory and linear regions
- [He et al., "Deep Residual Learning" (2015)](https://arxiv.org/abs/1512.03385) — Residual composition for very deep networks
