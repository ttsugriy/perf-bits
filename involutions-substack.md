# Involutions: The License to Update in O(1)

*Why Zobrist hashing updates a chessboard in constant time, and the property that makes incremental computation possible.*

---

Here's a puzzle from competitive chess programming:

**A chess engine evaluates millions of positions per second. Many are duplicates—reached via different move orders. How does it recognize positions it's already seen?**

The naive answer: hash the board and check a table.

But "hash the board" hides a cost. A chessboard has 64 squares, each potentially holding one of 12 piece types. Hashing from scratch is O(64) per position.

At millions of positions per second, that adds up.

The actual answer: **update the hash incrementally in O(1)**.

Move a knight from b1 to c3? Two operations. Not 64.

The technique is called Zobrist hashing. And it works because of a single mathematical property.

---

## The Property

An **involution** is a function that is its own inverse:

$$f(f(x)) = x$$

Apply it twice, you're back where you started.

The simplest example: XOR.

$$a \oplus a = 0$$

$$a \oplus 0 = a$$

XOR any value with itself, you get zero. XOR with zero, you get the original back. This means XOR is reversible—and the forward and reverse operations are identical.

This unlocks three capabilities:

- **Add = Remove**: Same operation to insert and delete
- **No history needed**: Don't track what was added when
- **O(1) updates**: Cost independent of total size

To modify a hash, you don't need to know how it was built—just what changed.

---

## Zobrist Hashing

Albert Zobrist invented this technique in 1970 for computer Go. It's now standard in every serious chess engine.

### Setup

Generate a table of random 64-bit numbers:

```python
import random

# One random number for each (piece, square) combination
zobrist_table = {}
for piece in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
    for square in range(64):
        zobrist_table[(piece, square)] = random.getrandbits(64)

# Additional random numbers for game state
zobrist_black_to_move = random.getrandbits(64)
zobrist_castling = [random.getrandbits(64) for _ in range(16)]
zobrist_en_passant = [random.getrandbits(64) for _ in range(8)]
```

These numbers are generated once at startup and never change.

### Initial Hash

To hash a position from scratch:

```python
def hash_position(board):
    h = 0
    for square in range(64):
        piece = board[square]
        if piece:
            h ^= zobrist_table[(piece, square)]

    if board.black_to_move:
        h ^= zobrist_black_to_move
    # ... castling rights, en passant, etc.

    return h
```

XOR together the random numbers for every piece on its square. O(n) where n is the number of pieces.

### Incremental Update

Here's where involutions pay off.

When a knight moves from b1 (square 1) to c3 (square 18):

```python
def update_hash_for_move(h, piece, from_sq, to_sq):
    h ^= zobrist_table[(piece, from_sq)]  # Remove piece from old square
    h ^= zobrist_table[(piece, to_sq)]    # Add piece to new square
    h ^= zobrist_black_to_move            # Toggle side to move
    return h
```

Three XOR operations. Constant time. The hash after the move is exactly what you'd get from hashing the new position from scratch.

**Why does this work?**

XOR is an involution. "Removing" a piece means XORing with the same value that "added" it. No separate undo operation needed.

```
hash = ... ^ zobrist[N][b1] ^ ...     # Knight on b1 contributes to hash

hash ^= zobrist[N][b1]                 # XOR same value → removes contribution
hash ^= zobrist[N][c3]                 # XOR new value → adds new contribution
```

The knight's contribution to the hash is completely replaced. The 62 other squares are unaffected.

---

## Try It Yourself

I built an interactive demo where you can move pieces and watch the XOR operations happen in real-time:

**[→ Interactive Zobrist Hashing Demo](https://ttsugriy.github.io/perf-bits/zobrist-demo.html)**

Notice: whether you move a pawn (1 XOR out, 1 XOR in) or capture a queen (2 XOR out, 1 XOR in), the operation count stays constant. The other 62 squares are never touched.

---

## Why XOR Specifically?

XOR isn't the only involution. Why not use subtraction? After all:

$$a - a = 0$$

Three reasons make XOR ideal for hashing:

### 1. Bitwise Independence

XOR operates on each bit independently. No carries, no overflow.

```
  10110011
^ 11001010
----------
  01111001
```

This means the random numbers can be generated for each bit position separately, maximizing entropy distribution.

### 2. Order Independence (Commutativity)

$$a \oplus b = b \oplus a$$

The hash doesn't depend on the order pieces were added. Two positions with the same pieces produce the same hash regardless of move history.

### 3. No Accumulation Bias

With addition/subtraction, values can grow unboundedly or cluster around certain ranges. XOR keeps the hash uniformly distributed across the full 64-bit range.

| Operation | Self-inverse? | Commutative? | Bounded? | Bit-independent? |
|-----------|---------------|--------------|----------|------------------|
| XOR | ✓ (a⊕a=0) | ✓ | ✓ | ✓ |
| Addition/Subtraction | ✗ (need inverse op) | ✓ | ✗ | ✗ |
| Multiplication | ✗ | ✓ | ✗ | ✗ |

XOR is unique: adding and removing use the *same* operation. With addition, you must subtract to undo—a different operation that requires knowing what was added.

---

## The Transposition Table

Zobrist hashing enables **transposition tables**—caches of previously evaluated positions.

```
1. e4 e5 2. Nf3 Nc6     →  position P
1. Nf3 Nc6 2. e4 e5     →  same position P
```

Different move orders, identical position. Same Zobrist hash.

When the engine encounters a position:

```python
def evaluate_with_cache(position, depth):
    h = position.zobrist_hash

    if h in transposition_table:
        cached = transposition_table[h]
        if cached.depth >= depth:
            return cached.score  # Skip expensive search

    score = expensive_minimax_search(position, depth)
    transposition_table[h] = Entry(score, depth)
    return score
```

At depth 20, a chess engine might explore 10⁹ positions. With a good transposition table, it might only evaluate 10⁷ unique positions. The rest are cache hits.

**O(1) hash updates make this practical.** If updating the hash cost O(64) per move, the overhead would dwarf the savings.

---

## Collision Reality

Zobrist hashing isn't cryptographically secure. Collisions happen.

With a 64-bit hash and 2³² positions, the birthday paradox predicts roughly one collision. In practice, chess engines search far more positions than that.

**Does it matter?**

Usually not. A collision means two different positions share a hash. The engine might:
- Return the wrong cached evaluation
- Prune a branch incorrectly

These are rare and the impact is statistical, not catastrophic. The engine plays slightly worse on rare occasions. Acceptable for a 100x speedup.

For applications where correctness matters absolutely, you'd verify the full position on cache hits—but that adds overhead.

---

## Beyond Chess: Rolling XOR

The same principle applies anywhere you need incremental hash updates.

**Sliding window over a byte stream:**

```python
from collections import deque

def rolling_xor_hash(window_size):
    h = 0
    buffer = deque()

    def add_byte(b):
        nonlocal h
        h ^= b
        buffer.append(b)
        if len(buffer) > window_size:
            old = buffer.popleft()
            h ^= old  # Remove old byte's contribution
        return h

    return add_byte
```

Each byte added, one byte removed. O(1) regardless of window size.

This is simpler than polynomial rolling hashes (Rabin-Karp), though it has weaker collision properties. For applications where speed matters more than collision resistance—like deduplication with verification—it's often sufficient.

---

## Beyond Chess: CRDTs

Conflict-free Replicated Data Types (CRDTs) enable distributed systems to merge state without coordination.

**XOR-based sets** exploit involutions:

```python
class XORSet:
    def __init__(self):
        self.hash = 0
        self.elements = set()

    def add(self, x):
        if x not in self.elements:
            self.hash ^= hash(x)
            self.elements.add(x)

    def remove(self, x):
        if x in self.elements:
            self.hash ^= hash(x)  # Same operation as add!
            self.elements.discard(x)
```

Two replicas can independently add and remove elements. Merging is deterministic because XOR is commutative and associative.

---

## Beyond Chess: The rsync Algorithm

Here's a different flavor of incremental hashing—one that synchronizes files across a network.

**The problem**: You have a 1GB file locally. The server has a slightly modified version. Sending the entire file wastes bandwidth. But you don't know *which* bytes changed.

**rsync's insight**: Use rolling checksums to find matching blocks without transferring them.

### The Protocol

1. **Receiver** (has old file): Split into blocks, compute weak checksum (Adler-32) + strong checksum (MD5) for each block, send list to sender

2. **Sender** (has new file): Compute rolling checksum at every byte offset, match against list

3. **Result**: Send only "match block B2" references + the changed bytes

The magic is computing checksums at *every byte offset* efficiently.

### The Rolling Checksum

rsync uses a variant of Adler-32. For a window of bytes $[a_1, a_2, \ldots, a_n]$:

$$A = \sum_{i=1}^{n} a_i \mod M$$

$$B = \sum_{i=1}^{n} (n - i + 1) \cdot a_i \mod M$$

When the window slides one byte (drop $a_1$, add $a_{n+1}$):

$$A' = A - a_1 + a_{n+1}$$

$$B' = B - n \cdot a_1 + A'$$

**O(1) per byte**, regardless of window size. No need to re-sum all n bytes.

### Involutions vs. Rolling Updates

Zobrist and rsync both achieve O(1) updates, but through different mechanisms:

| Technique | Mathematical Basis | Update Operation |
|-----------|-------------------|------------------|
| Zobrist | XOR involution: $a \oplus a = 0$ | Toggle element membership |
| rsync | Additive sliding: $A' = A - a_{old} + a_{new}$ | Slide window by one position |

Zobrist uses XOR's self-inverse property—add and remove are the same operation.

rsync uses modular arithmetic's linearity—you can subtract the old contribution and add the new one separately.

Both exploit structure to avoid recomputation. The algebra differs, but the principle is identical: **incremental updates beat full recalculation**.

---

## The Counter-Example: Cryptographic Hashes

SHA-256 is deliberately **not** invertible.

```python
h1 = sha256(data)
# There is no sha256_remove(h1, partial_data)
```

If you change one byte, you must rehash everything. That's the point—cryptographic hashes are designed so you can't deduce anything about the input from the output.

| Hash Type | Incremental? | Collision-resistant? | Use Case |
|-----------|--------------|---------------------|----------|
| Zobrist (XOR) | ✓ O(1) | Weak | Game trees, caches |
| Rolling polynomial | ✓ O(1) | Moderate | String matching |
| SHA-256 | ✗ O(n) | Strong | Security, integrity |
| Merkle tree | ✓ O(log n) | Strong | Blockchain, git |

Merkle trees are an interesting middle ground: O(log n) updates with cryptographic security. You sacrifice the O(1) of XOR but gain collision resistance.

---

## Designing for Involutions

When you need incremental hash updates, ask:

**1. What's changing?**
- Single element: Zobrist-style XOR
- Sliding window: Rolling hash
- Tree structure: Merkle tree

**2. How bad are collisions?**
- Recoverable (cache miss is fine): XOR
- Needs verification: XOR + full comparison
- Catastrophic (security): Merkle or rehash

**3. What operations do you need?**
- Add only: Many options
- Add and remove: Need involution
- Order matters: Can't use commutative XOR

The involution property isn't just an optimization. It determines what operations are even possible in O(1).

---

## The Takeaway

Zobrist hashing achieves O(1) updates because XOR is an involution—its own inverse.

**Adding and removing are the same operation.** Move a piece? XOR out the old square, XOR in the new square. The hash updates without knowing anything about the other 62 squares.

This single property—$f(f(x)) = x$—enables:
- Chess engines to cache millions of positions
- Distributed systems to merge state without coordination
- Streaming algorithms to maintain hashes over sliding windows

The involution came first. The algorithms followed.

When you need incremental updates, don't reach for clever data structures.

Reach for the algebra.

---

*See also: [Associativity: The One Property That Makes FlashAttention Possible](https://softwarebits.substack.com/p/the-one-property-that-makes-flashattention)*

*Full article with interactive demo: [ttsugriy.github.io/perf-bits/involutions.html](https://ttsugriy.github.io/perf-bits/involutions.html)*

---

## Further Reading

- [Zobrist, "A New Hashing Method with Application for Game Playing" (1970)](https://research.cs.wisc.edu/techreports/1970/TR88.pdf) — The original paper
- [Chessprogramming Wiki: Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing) — Comprehensive reference
- [Tridgell & Mackerras, "The rsync Algorithm" (1996)](https://rsync.samba.org/tech_report/) — Efficient file synchronization
- [Shapiro et al., "Conflict-free Replicated Data Types" (2011)](https://hal.inria.fr/inria-00609399/document) — CRDTs and commutative operations
