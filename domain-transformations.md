---
layout: single
title: "Domain Transformations: The Art of Finding Easier Spaces"
excerpt: "Why logarithms prevent underflow, why Fourier speeds up convolutions, and how choosing the right space makes hard problems tractable"
toc: true
toc_sticky: true
math: true
---

*Subscribe to [Software Bits](https://softwarebits.substack.com/) to get new articles in your inbox.*

---

Here's a trick that appears everywhere in numerical computing:

**Don't solve the hard problem. Transform it into an easy one.**

Multiplying a thousand small probabilities? You'll underflow to zero. But *add their logarithms*, and everything works.

Convolving two signals of length n? That's O(n²). But *multiply in Fourier space*, and it's O(n log n).

These aren't clever hacks. They're instances of a general principle: **the right domain makes computation tractable.**

---

## The Principle

A domain transformation moves computation from one space to another:

$$\text{Hard problem in Space A} \xrightarrow{\text{transform}} \text{Easy problem in Space B}$$

The pattern:

1. Transform input to the easier domain
2. Compute in that domain
3. Transform back (if needed)

This is worthwhile when:
$$\text{cost(transform)} + \text{cost(easy computation)} + \text{cost(inverse)} < \text{cost(hard computation)}$$

Often dramatically so.

---

## Logarithms: Multiplication to Addition

The most common domain transformation in ML.

### The Problem

You're computing the probability of a sequence:

$$P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_{<i})$$

Each factor is less than 1. Multiply 1000 of them:

```python
>>> 0.1 ** 1000
0.0  # Underflow
```

The true answer isn't zero—it's just smaller than the smallest representable float.

### The Solution

Work in log-space:

$$\log P = \sum_{i=1}^{n} \log P(x_i | x_{<i})$$

Multiplication becomes addition. Products of tiny numbers become sums of negative numbers.

```python
>>> import math
>>> 1000 * math.log(0.1)
-2302.58...  # No underflow
```

| Operation | Linear Space | Log Space |
|-----------|--------------|-----------|
| Multiply | Product (underflows) | Sum (stable) |
| Divide | Quotient | Difference |
| Power | Exponentiation | Multiplication |

This is why every language model outputs log-probabilities. Why Bayesian inference uses log-likelihoods. Why HMMs compute in log-space.

**Logarithms don't just prevent underflow. They turn an unstable operation into a stable one.**

---

## The Log-Sum-Exp Trick

But what if you need to *add* probabilities, not multiply them?

For mutually exclusive events:

$$P(A \text{ or } B) = P(A) + P(B)$$

In log-space, this becomes:

$$\log(e^a + e^b)$$

where $a = \log P(A)$ and $b = \log P(B)$. This comes up constantly—marginalizing over hidden states in HMMs, summing over paths in CTC, computing partition functions.

### The Problem

```python
>>> import math
>>> a, b = -1000, -1001
>>> math.log(math.exp(a) + math.exp(b))
# math.exp(-1000) = 0.0 (underflow)
# Result: undefined
```

Exponentiating large negative numbers underflows. We're back where we started.

### The Solution

Factor out the maximum:

$$\log(e^a + e^b) = \max(a, b) + \log(1 + e^{-|a-b|})$$

The exponent $-|a-b|$ is always ≤ 0, so $e^{-|a-b|} \leq 1$. No overflow, and the underflowed term is added to 1, so we don't lose precision.

```python
def log_sum_exp(a, b):
    max_val = max(a, b)
    return max_val + math.log(math.exp(a - max_val) + math.exp(b - max_val))
```

For vectors:

$$\text{logsumexp}(x) = \max(x) + \log \sum_i \exp(x_i - \max(x))$$

This is exactly what softmax uses internally:

$$\text{softmax}(x)_i = \exp(x_i - \text{logsumexp}(x))$$

**The log-sum-exp trick keeps computation in log-space while handling addition.**

---

## Fourier Transform: Convolution to Multiplication

Another profound domain transformation.

### The Problem

Convolution in the spatial domain:

$$(f * g)[n] = \sum_{m} f[m] \cdot g[n-m]$$

For signals of length N, this is O(N²)—N outputs, each summing over N terms.

### The Convolution Theorem

The Fourier transform has a remarkable property:

$$\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$$

Convolution in space = multiplication in frequency.

The algorithm:

```python
def fast_convolve(f, g):
    F_f = fft(f)        # O(n log n)
    F_g = fft(g)        # O(n log n)
    F_result = F_f * F_g  # O(n) - element-wise
    return ifft(F_result) # O(n log n)
```

Total: O(n log n) instead of O(n²).

| Domain | Operation | Complexity |
|--------|-----------|------------|
| Spatial | Convolution | O(n²) |
| Frequency | Element-wise multiply | O(n) |
| Transform | FFT/IFFT | O(n log n) |

For large kernels, the transform cost is negligible. For small kernels (3×3, 5×5), direct convolution is faster—the transform overhead dominates.

In practice, modern deep learning frameworks use highly optimized direct convolution (cuDNN) even for medium-sized kernels, because GPU implementations are so fast. But FFT-based convolution remains important in signal processing and for very large kernels.

---

## Why Transformers Don't Use FFT

Attention looks like it should benefit from FFT:

$$\text{Attention} = \text{softmax}(QK^T)V$$

The $QK^T$ multiplication is O(n²d) for sequence length n.

But there's a catch: **attention isn't convolution**.

Convolution is translation-equivariant—shift the input, shift the output. The kernel slides uniformly.

Attention is content-based—each position attends differently based on *what's there*, not where it is.

The Fourier transform exploits the structure of convolution. Attention doesn't have that structure.

(Some efficient attention methods like Performers and Linear Attention do use kernel approximations—but they're approximating attention, not computing it exactly via FFT.)

---

## Embeddings: Sparse to Dense

A different kind of domain transformation.

### The Problem

Words are categorical—there's no natural arithmetic on them.

"king" - "man" + "woman" = ???

With one-hot encoding, words are orthogonal vectors in a 50,000-dimensional space. No similarity, no structure.

### The Solution

Learn a dense embedding:

$$\text{word} \xrightarrow{\text{embed}} \mathbb{R}^{d}$$

Now:

$$\text{embed}(\text{"king"}) - \text{embed}(\text{"man"}) + \text{embed}(\text{"woman"}) \approx \text{embed}(\text{"queen"})$$

The transformation creates structure that wasn't there before.

| Domain | Dimensionality | Structure |
|--------|---------------|-----------|
| One-hot | 50,000 (sparse) | None |
| Embedding | 256-1024 (dense) | Semantic similarity |

This isn't just dimensionality reduction. It's finding a space where the hard problem (word relationships) becomes easy (vector arithmetic).

---

## The Kernel Trick: Linear in a Lifted Space

Sometimes the solution isn't a lower-dimensional space—it's a higher one.

### The Problem

Data isn't linearly separable:

```
         ○   ○   ○
       ○           ○
     ○    ●  ●  ●    ○
     ○    ●  ●  ●    ○
     ○    ●  ●  ●    ○
       ○           ○
         ○   ○   ○

  No linear boundary separates ● from ○.
```

The inner class is surrounded. No line can divide them.

### The Solution

Map to a higher-dimensional space where it *is* linearly separable:

$$\phi: \mathbb{R}^d \to \mathbb{R}^D \quad \text{where } D > d$$

For the concentric pattern, add just one feature: $z = x^2 + y^2$ (squared distance from center). Now the inner class has small $z$, the outer class has large $z$. A horizontal plane separates them.

The kernel trick computes dot products in the lifted space *without explicitly computing the transformation*:

$$K(x, y) = \langle \phi(x), \phi(y) \rangle$$

For the RBF kernel:

$$K(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

This implicitly works in an *infinite*-dimensional space, but you never compute it directly.

**The kernel trick is a domain transformation you don't have to pay for.**

---

## Spectral Graph Methods

Graph Laplacians transform graph problems into linear algebra.

### The Problem

Clustering nodes in a graph based on connectivity. Community detection. Graph partitioning.

These seem like discrete, combinatorial problems.

### The Solution

Compute the graph Laplacian:

$$L = D - A$$

where D is the degree matrix and A is the adjacency matrix.

The eigenvectors of L embed nodes into Euclidean space:

$$\text{node}_i \to (v_1(i), v_2(i), \ldots, v_k(i))$$

In this space, k-means clustering solves the graph partitioning problem.

A discrete optimization problem becomes a tractable eigenvector computation.

---

## Numerical Stability: A Recurring Theme

Many domain transformations exist for numerical reasons:

| Problem | Unstable Form | Stable Form |
|---------|--------------|-------------|
| Product of probabilities | $\prod p_i$ | $\sum \log p_i$ |
| Softmax | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $\frac{e^{x_i - \max}}{\sum e^{x_j - \max}}$ |
| Variance | $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ | Welford's online algorithm |
| Matrix inverse | Direct inversion | QR/SVD decomposition |

Each is a domain transformation that trades mathematical equivalence for numerical stability.

**Equivalent in exact arithmetic. Different in floating-point.**

---

## When to Look for a Transform

The pattern recurs:

1. **Multiplication → Addition**: Use logarithms
2. **Convolution → Element-wise**: Use Fourier
3. **Non-linear → Linear**: Use kernel methods
4. **Discrete → Continuous**: Use spectral methods
5. **High-dimensional sparse → Low-dimensional dense**: Use embeddings

The skill is recognizing when your hard problem has an easier dual.

Questions to ask:

- **What's expensive or unstable?** Products of small numbers? Convolution? Non-linear optimization?
- **Is there a standard transform?** Log, Fourier, Laplace, embeddings?
- **Does the transform preserve what you need?** Some transforms lose information.
- **Is the round-trip worth it?** Transform cost + easy computation < hard computation?

---

## The Meta-Lesson

The previous articles in this series covered:

- **Associativity**: Enables chunking, parallelization, streaming
- **Commutativity**: Enables reordering, permutation invariance
- **Linearity**: Enables batching, gradient accumulation

Domain transformations are different. They're not about properties of operations—they're about **choosing the right space**.

But the connection is deep: transforms often work because they reveal simpler structure.

- Fourier reveals that convolution is *pointwise* in frequency space
- Logarithms reveal that multiplication is *additive* in log space
- Embeddings reveal that semantic relationships are approximately *linear* in embedding space

**The transform finds a space where the problem has the structure you need.**

---

## The Takeaway

Don't solve the hard problem directly. Ask: **is there an easier space?**

- Products underflow → work in log space
- Convolution is O(n²) → work in frequency space
- Data isn't linearly separable → work in kernel space

The transform might cost something. But if the computation in the new space is dramatically cheaper, you win.

This is computational judo: using structure to redirect effort. The best algorithms don't fight the problem—they find the space where the problem solves itself.

---

*Next in this series: [Sparsity: The License to Skip](sparsity.html)—why ignoring most of your neural network is the key to efficiency.*

---

## Further Reading

- [Numerical Recipes, Ch. 12-13](http://numerical.recipes/) — FFT and spectral methods
- [Murphy, "Probabilistic Machine Learning"](https://probml.github.io/pml-book/) — Log-space computation throughout
- [Scholkopf & Smola, "Learning with Kernels"](https://mitpress.mit.edu/9780262536578/) — Kernel methods and feature spaces
- [Von Luxburg, "A Tutorial on Spectral Clustering"](https://arxiv.org/abs/0711.0189) — Graph Laplacians and embeddings
