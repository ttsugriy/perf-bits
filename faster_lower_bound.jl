### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 93c5d31b-ea33-4d87-9e41-c4bc15b85022
using Base.Order

# ╔═╡ af5fb256-4ab5-4914-92a0-562c6e9b9b9e
using Base.Sort

# ╔═╡ 22224627-44f6-4978-8aca-c85cc3f7fef3
using BenchmarkTools

# ╔═╡ d07ed199-931a-42df-9dd6-9f4b976cb67e
using Test

# ╔═╡ 8776ad24-5bdb-49a2-8319-69473093a0bd
md"""
# Speeding up Julia's searchsortedfirst.
## Or even standard libraries can be improved.

Most of my articles so far focused on performance problems I've actually experienced, but I had to simplify or edit them to a point where they no longer look like something that can happen in real world.

To make things slightly more interesting, for this article, I've decided to take a look at actual Julia code in its Sort package. Since, `lower_bound` is one of the algorithms I used most frequently in C++, I wanted to see how it's implemented in Julia.

Apart from its name `searchsortedfirst` it looks [as expected](https://github.com/JuliaLang/julia/blob/master/base/sort.jl#L172-L187)
```julia
# index of the first value of vector a that is greater than or equal to x;
# returns lastindex(v)+1 if x is greater than all values in v.
function searchsortedfirst(v::AbstractVector, x, lo::T, hi::T, o::Ordering)::keytype(v) where T<:Integer
    u = T(1)
    lo = lo - u
    hi = hi + u
    @inbounds while lo < hi - u
        m = midpoint(lo, hi)
        if lt(o, v[m], x)
            lo = m
        else
            hi = m
        end
    end
    return hi
end
```

Some of my initial observations included:
- the search spaced is extended to $[lo-1, hi+1]$
- each iteration computes `midpoint`
- the range is narrowed by setting `lo` or `hi` to `m`.

So what can we do instead?

- let's call the returned value $x$ and based on the docs $x \in [0, lastindex(v) + 1]$, so the range we're interested in is $[lo, hi+1]$.
- the `midpoint`'s implementation does a classical overflow-safe
```julia
# This implementation of `midpoint` is performance-optimized but safe
# only if `lo <= hi`.
midpoint(lo::T, hi::T) where T<:Integer = lo + ((hi - lo) >>> 0x01)
midpoint(lo::Integer, hi::Integer) = midpoint(promote(lo, hi)...)
``` which we can get rid of by tracking explicit range length
- on etablishing that the `v[m]` is smaller than `x` we can exclude `m` from the range and narrow it down to $[m + 1, hi]$ instead.

Putting all of these things together yeilds an implementation below
"""

# ╔═╡ d870da60-064f-11ed-2744-810f2b87b124
function mysearchsortedfirst(v::AbstractVector, x, lo::T, hi::T, o::Ordering)::keytype(v) where T<:Integer
    hi = hi + T(1)
	len = hi - lo
    @inbounds while len != 0
		half_len = len >>> 0x01
        m = lo + half_len
        if lt(o, v[m], x)
            lo = m + 1
			len -= half_len + 1
        else
            hi = m
			len = half_len
        end
    end
    return lo
end

# ╔═╡ 12c0dfc8-5720-46d8-9ccd-7a83ab2d63c0
md"""
Is it better though? Yes, according to benchmarks below.

Hopefully [these changes](https://github.com/JuliaLang/julia/pull/46151) are going to be accepted and land in Julia's main repository.
"""

# ╔═╡ 094a85f5-efc1-48b1-8efd-0bfc9dcba7c2
xs = sort(randn(10000))

# ╔═╡ 7b74ded7-5258-4db5-9203-2a8a390733e9
x = rand(xs)

# ╔═╡ 50fa7c82-f64e-4c72-8fdf-54bbb7330574
mysearchsortedfirst(xs, x, firstindex(xs), lastindex(xs), Forward)

# ╔═╡ bedb0d1a-d639-407b-8e73-8837ddffb500
searchsortedfirst(xs, x, firstindex(xs), lastindex(xs), Forward)

# ╔═╡ 6537fd8d-d998-4bc6-a238-c4e2383b195f
@benchmark searchsortedfirst(xs, x, firstindex(xs), lastindex(xs), Forward)

# ╔═╡ b531a66c-5746-4710-97dd-db1dd7984065
@benchmark mysearchsortedfirst(xs, x, firstindex(xs), lastindex(xs), Forward)

# ╔═╡ 18bc18eb-46f7-4e86-a5c7-a7e33faab906
# @code_llvm mysearchsortedfirst(xs, x, firstindex(xs), lastindex(xs), Forward)

# ╔═╡ a33ea131-e098-4044-a481-a4d6a03de98c
for t in xs
	@test mysearchsortedfirst(xs, x, firstindex(xs), lastindex(xs), Forward) == searchsortedfirst(xs, x, firstindex(xs), lastindex(xs), Forward)
end

# ╔═╡ 54a6c1e8-f800-4598-9562-ac49db77bdb4
@test mysearchsortedfirst(xs, xs[1] - 1, firstindex(xs), lastindex(xs), Forward) == 1

# ╔═╡ 99af18ba-bcae-46f5-a2ab-c20ccd643e78
@test mysearchsortedfirst(xs, xs[lastindex(xs)] + 1, firstindex(xs), lastindex(xs), Forward) == lastindex(xs) + 1

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

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

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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
# ╟─8776ad24-5bdb-49a2-8319-69473093a0bd
# ╠═93c5d31b-ea33-4d87-9e41-c4bc15b85022
# ╠═d870da60-064f-11ed-2744-810f2b87b124
# ╟─12c0dfc8-5720-46d8-9ccd-7a83ab2d63c0
# ╠═af5fb256-4ab5-4914-92a0-562c6e9b9b9e
# ╠═094a85f5-efc1-48b1-8efd-0bfc9dcba7c2
# ╠═7b74ded7-5258-4db5-9203-2a8a390733e9
# ╠═50fa7c82-f64e-4c72-8fdf-54bbb7330574
# ╠═bedb0d1a-d639-407b-8e73-8837ddffb500
# ╠═22224627-44f6-4978-8aca-c85cc3f7fef3
# ╠═6537fd8d-d998-4bc6-a238-c4e2383b195f
# ╠═b531a66c-5746-4710-97dd-db1dd7984065
# ╠═18bc18eb-46f7-4e86-a5c7-a7e33faab906
# ╠═d07ed199-931a-42df-9dd6-9f4b976cb67e
# ╠═a33ea131-e098-4044-a481-a4d6a03de98c
# ╠═54a6c1e8-f800-4598-9562-ac49db77bdb4
# ╠═99af18ba-bcae-46f5-a2ab-c20ccd643e78
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
