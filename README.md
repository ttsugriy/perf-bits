# Perf Bits

A collection of interactive [Pluto.jl](https://plutojl.org/) notebooks exploring performance optimization techniques across different programming languages.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [fast_to_upper.jl](fast_to_upper.jl) | Binary trickery for implementing a fast `toupper` using a single bit operation |
| [faster_fibonacci.jl](faster_fibonacci.jl) | Using semigroups and matrix exponentiation to compute Fibonacci numbers in O(log n) time |
| [faster_lower_bound.jl](faster_lower_bound.jl) | Optimizing Julia's `searchsortedfirst` (equivalent to C++'s `lower_bound`) |
| [random_vs_randint.jl](random_vs_randint.jl) | Why Python's `random.random()` is 5x faster than `random.randint()` |
| [sprintf_vs_to_chars.jl](sprintf_vs_to_chars.jl) | Comparing `sprintf` vs `std::to_chars` for numeric-to-string conversion |
| [virtual-functions-strike-again.jl](virtual-functions-strike-again.jl) | Virtual dispatch overhead and the `-fstrict-vtable-pointers` optimization |

## Requirements

- [Julia](https://julialang.org/) 1.7+
- [Pluto.jl](https://github.com/fonsp/Pluto.jl)

## Getting Started

1. Install Julia from [julialang.org](https://julialang.org/downloads/)

2. Install Pluto.jl:
   ```julia
   using Pkg
   Pkg.add("Pluto")
   ```

3. Launch Pluto and open any notebook:
   ```julia
   using Pluto
   Pluto.run()
   ```

4. Navigate to a notebook file (e.g., `faster_fibonacci.jl`) to view and interact with it.

## License

MIT License - see [LICENSE](LICENSE) for details.
