### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ bd423c52-15fe-11ed-0966-df233e03ce69
md"""
# Faster Fibonacci
## Or the hidden power of semigroups.

At this point computing `n`th Fibonacci number is practically a "hello world" of any course on recursion or dynamic programming. As such I won't go into the details as to what's so interesting about them and why neither recursive nor dynamic programming approaches are optimal for their computation. At the same time, to make sure that we are all on the same page, it's a series of numbers $$0, 1, 1, 2, 3, 5, 8, ...$$ where every number is a sum of its 2 predecessors. This definition lands itself naturally to a recursive formula $$F_n = F_{n-2} + F_{n-1}$$ which in turn can be trivially turned into a recursive function:
"""

# ╔═╡ f312a216-25e4-4a88-ba38-9ff6b266b12f
function fib_rec(n)
	if n < 2
		return n
	end
	fib_rec(n-2) + fib_rec(n-1)
end

# ╔═╡ e7bc1129-87e3-46e1-9fe7-4476cbc6be3a
map(fib_rec, 0:10)

# ╔═╡ 4b86c389-8e0d-467a-af7d-5425ec6f7272
md"""
Enough has already been written about why it's very inefficient and its easy to understand why from the call chain
![call chain](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fi.stack.imgur.com%2F59Rpw.png&f=1&nofb=1)
There are lots of repeated computations that result in unsutisfactory exponential runtime ($O(\varphi^n))$, where $\varphi$ is the golden ratio, to be exact).
That's where most lectures go into ways to use dynamic programming techniques like memoization or bottom up tabulation to reach much better $(O(N))$ time complexity.
"""

# ╔═╡ d47c1b03-dc1b-47c7-a92e-1f4c8e405988
function fib_bu(n)
	n_2, n_1 = 0, 1
	for i in 1:n
		n_2, n_1 = n_1, n_2 + n_1
	end
	n_2
end

# ╔═╡ 41d4489a-91b2-420f-b77d-134d680fcfbb
map(fib_bu, 0:10)

# ╔═╡ a5990b0c-b5bb-42bd-82c8-2c25b9d50637
md"""
And yes, it does matter in practice.
"""

# ╔═╡ bb895a21-93d3-4e41-b49a-11eec3c5894c
import BenchmarkTools

# ╔═╡ a6336da1-64bd-4ca2-9f7c-538b14cba139
BenchmarkTools.@benchmark fib_rec(20)

# ╔═╡ 1ba05c1c-461a-4b65-a1e9-5be66a14da41
BenchmarkTools.@benchmark fib_bu(20)

# ╔═╡ 43763f6a-9d10-4217-911d-1fed80ddcf9e
md"""
Ok, but what does it have to do with semigroups? Nothing, not yet, at least. Let's make an interesting observation:
```math
\begin{bmatrix}
F_0 \\
F_1
\end{bmatrix}
\times
\begin{bmatrix}
  0 & 1\\
  1 & 1
\end{bmatrix}
=
\begin{bmatrix}
F_1\\
F_0 + F_1
\end{bmatrix}
\equiv
\begin{bmatrix}
F_1\\
F_2
\end{bmatrix}
```
By repeating this process, we can compute any number of want.
```math
\begin{bmatrix}
0\\
1
\end{bmatrix}
\times
\begin{bmatrix}
0 & 1\\
1 & 1
\end{bmatrix}
\times
\dots
\times
\begin{bmatrix}
0 & 1\\
1 & 1
\end{bmatrix}
\equiv
\begin{bmatrix}
0\\
1
\end{bmatrix}
\times
\begin{bmatrix}
0 & 1\\
1 & 1
\end{bmatrix}^n
=
\begin{bmatrix}
F_n \\
F_{n+1}
\end{bmatrix}
```
Well, looks like we've successfully managed to replace scalar multiplications with more expensive matrix multiplications. Doesn't look like a win, does it?

And now it's finally time for some definitions:

> [Associative](https://mathworld.wolfram.com/Associative.html): Three elements $x$, $y$ and $z$ of a set $S$ are said to be associative under a binary operation $*$ if they satisfy $x * (y * z) = (x * y) * z$.

While this property may not seem significant, this property is what enables divide and conquer algorithms, which we'll use later on.

> [Semigroup](https://mathworld.wolfram.com/Semigroup.html): A mathematical object defined for a set and a binary operator in which the multiplication operation is associative.

And guess what, matrix multiplication for square matrices is associative, so we have none other than a semigroup!
What's exciting about this fact is much easier to understand after looking at [Haskell's `Semigroup` typeclass](https://wiki.haskell.org/Data.Semigroup):
```haskell
class Semigroup a where
    (<>) :: a -> a -> a
    ...
    stimes :: Integral b => b -> a -> a
```
Have you noticed `stimes`? It's not really part of what semigroups are, but associativity makes it possible to implement repeated application of the semigroup operation faster than in $O(N)$ time. Haskell's default implementation is probably not the most [elegant](https://hackage.haskell.org/package/base-4.16.0.0/docs/src/Data.Semigroup.Internal.html#stimesDefault):
```haskell
stimesDefault :: (Integral b, Semigroup a) => b -> a -> a
stimesDefault y0 x0
  | y0 <= 0   = errorWithoutStackTrace "stimes: positive multiplier expected"
  | otherwise = f x0 y0
  where
    f x y
      | even y = f (x <> x) (y `quot` 2)
      | y == 1 = x
      | otherwise = g (x <> x) (y `quot` 2) x        -- See Note [Half of y - 1]
    g x y z
      | even y = g (x <> x) (y `quot` 2) z
      | y == 1 = x <> z
      | otherwise = g (x <> x) (y `quot` 2) (x <> z) -- See Note [Half of y - 1]
```
but it's pretty much a well known fast exponentiation algorithm
```math
x^n =
\begin{cases}
1 &  \text{if n = 0} \\
x * (x^{(n-1)/2})^2 & \text{if n is odd} \\
(x^{n/2})^2 & \text{if n is even}
\end{cases}
```
which has a logarithmic complexity.
"""

# ╔═╡ bf2b2b38-a0d5-42f9-89fe-72423f339749
function fib_fast(n)
	@inbounds ([0 1; 1 1] ^ n * [0; 1])[1]
end

# ╔═╡ 66ad0db3-0571-4e7a-93c3-6e709c9ae5e4
map(fib_fast, 0:10)

# ╔═╡ 494c4da5-a1f4-43de-9ef0-017bcb8671a3
BenchmarkTools.@benchmark fib_fast(20)

# ╔═╡ 0f5e361c-4473-41ef-9c16-c43e77b18ece
md"""
Oh no, looks like all of our effort resulted in ~600X regression over our iterative implementation! What went wrong?
In theory, constant factors don't matter, but in practice... Extra complexity for dealing with matrices adds so much overhead that we have to go to much larger numbers to finally reap the benefits of logarithmic complexity.
"""

# ╔═╡ 5823dea2-153d-41bf-ae31-653576663239
BenchmarkTools.@benchmark fib_fast(10000)

# ╔═╡ cfc3db01-01ba-4b86-848e-c99e5190e095
BenchmarkTools.@benchmark fib_bu(10000)

# ╔═╡ 22f5de2f-287b-4935-b60e-a64cb3020e49
md"""
And sure enough, for 10000, matrix version is ~10X faster than $O(N)$ iterative implementation.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"

[compat]
BenchmarkTools = "~1.3.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─bd423c52-15fe-11ed-0966-df233e03ce69
# ╠═f312a216-25e4-4a88-ba38-9ff6b266b12f
# ╠═e7bc1129-87e3-46e1-9fe7-4476cbc6be3a
# ╟─4b86c389-8e0d-467a-af7d-5425ec6f7272
# ╠═d47c1b03-dc1b-47c7-a92e-1f4c8e405988
# ╠═41d4489a-91b2-420f-b77d-134d680fcfbb
# ╠═a5990b0c-b5bb-42bd-82c8-2c25b9d50637
# ╠═bb895a21-93d3-4e41-b49a-11eec3c5894c
# ╠═a6336da1-64bd-4ca2-9f7c-538b14cba139
# ╠═1ba05c1c-461a-4b65-a1e9-5be66a14da41
# ╟─43763f6a-9d10-4217-911d-1fed80ddcf9e
# ╠═bf2b2b38-a0d5-42f9-89fe-72423f339749
# ╠═66ad0db3-0571-4e7a-93c3-6e709c9ae5e4
# ╠═494c4da5-a1f4-43de-9ef0-017bcb8671a3
# ╟─0f5e361c-4473-41ef-9c16-c43e77b18ece
# ╠═5823dea2-153d-41bf-ae31-653576663239
# ╠═cfc3db01-01ba-4b86-848e-c99e5190e095
# ╟─22f5de2f-287b-4935-b60e-a64cb3020e49
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
