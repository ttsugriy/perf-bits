### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 21ffb7ae-0afe-11ed-295b-0f2473f2c9bb
md"""
# Generating random integers in Python.
## Or don't make assumptions about performance based on API.

Even though we love computers for their determinism, random number generation is a surprisingly common task used in a wide range of contexts - from quick sort partitioning to Monte Carlo simulations. As such, no wonder, Python comes with an entire [module](https://docs.python.org/3/library/random.html) dedicated to working with random numbers. So generating a random number is as simple as
```python
int(random.random() * 129)
```
Well, it’s not the most beautiful line of code, but it does the job, so time to check it in and move on.

Some time passes and we accidentally discover a nicer version using [random.randint](https://docs.python.org/3/library/random.html#random.randint)
```python
random.randint(0, 128)
```
Much better! Let's push this improved version to production... Strange, we start getting complaints about performance regression. How could that be? Let's take a look:
```
In [11]: %timeit int(random.random() * 129)
209 ns ± 1.88 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

In [12]: %timeit random.randint(0, 128)
1.04 µs ± 3.93 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
```
Oh no, it’s ~5X slower! How is it possible? Let’s take a look at the [source](https://github.com/python/cpython/blob/204946986feee7bc80b233350377d24d20fcb1b8/Modules/_randommodule.c#L157-L181)
```c
static PyObject *
_random_Random_random_impl(RandomObject *self)
/*[clinic end generated code: output=117ff99ee53d755c input=afb2a59cbbb00349]*/
{
    uint32_t a=genrand_uint32(self)>>5, b=genrand_uint32(self)>>6;
    return PyFloat_FromDouble((a*67108864.0+b)*(1.0/9007199254740992.0));
}
```
So `random.random` is implemented natively in C, but what about `random.randint`? It turns out it's [implemented in Python](https://github.com/python/cpython/blob/a2fbc511985f77c16c0f4a6fc6d3da9ab81a86b7/Lib/random.py#L287-L332):
```python
    def randrange(self, start, stop=None, step=_ONE):
        \"""Choose a random item from range(stop) or range(start, stop[, step]).
        Roughly equivalent to ``choice(range(start, stop, step))`` but
        supports arbitrarily large ranges and is optimized for common cases.
        \"""

        # This code is a bit messy to make it fast for the
        # common case while still doing adequate error checking.
        istart = _index(start)
        if stop is None:
            # We don't check for "step != 1" because it hasn't been
            # type checked and converted to an integer yet.
            if step is not _ONE:
                raise TypeError("Missing a non-None stop argument")
            if istart > 0:
                return self._randbelow(istart)
            raise ValueError("empty range for randrange()")

        # Stop argument supplied.
        istop = _index(stop)
        width = istop - istart
        istep = _index(step)
        # Fast path.
        if istep == 1:
            if width > 0:
                return istart + self._randbelow(width)
            raise ValueError(f"empty range in randrange({start}, {stop})")

        # Non-unit step argument supplied.
        if istep > 0:
            n = (width + istep - 1) // istep
        elif istep < 0:
            n = (width + istep + 1) // istep
        else:
            raise ValueError("zero step for randrange()")
        if n <= 0:
            raise ValueError(f"empty range in randrange({start}, {stop}, {step})")
        return istart + istep * self._randbelow(n)

    def randint(self, a, b):
        \"""Return random integer in range [a, b], including both end points.
        \"""

        return self.randrange(a, b+1)
```
As such, despite having an API that looks ideal for generating random integers, unfortunately its implementation leaves a significant performance gap. I normally recommend using the most idiomatic APIs unless there is a very strong reason not to, but ultimately the choice should be made based on business priorities.

Btw, since I'm writing this notebook in Julia, why not measure its performance for generating a random integer?
"""

# ╔═╡ 3cf6f73c-5e51-4101-a51d-dbd897f192fb
import BenchmarkTools

# ╔═╡ 28145330-430e-4288-9cac-8b182f21b7df
BenchmarkTools.@benchmark rand(0:128)

# ╔═╡ 55be1fe2-1920-4499-b0ec-b6e37f954713
md"""
That's fast! No wonder Julia is gradually finding its way into the hearts of former Python fans. Also, in case you're wondering you can see the native code produced for this expression below.
"""

# ╔═╡ 4bc8c7ff-07be-4ce9-9685-82d069a6e1b7
@code_native rand(0:128)

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
# ╟─21ffb7ae-0afe-11ed-295b-0f2473f2c9bb
# ╠═3cf6f73c-5e51-4101-a51d-dbd897f192fb
# ╠═28145330-430e-4288-9cac-8b182f21b7df
# ╟─55be1fe2-1920-4499-b0ec-b6e37f954713
# ╠═4bc8c7ff-07be-4ce9-9685-82d069a6e1b7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
