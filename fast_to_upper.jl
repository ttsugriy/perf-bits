### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 73267881-56f3-4a04-a96e-d2f62bda61e4
md"""
# Faster toupper implementation.
## Or binary trickery for the win.

Even though I'm not a fan of strings, sometimes I have to get over them and find ways to minimize their damage. Recently I've discovered an interesting way to implement `toupper` using a single bit operation.

Let's take a look at the binary representation of the lowercase letters:
"""

# ╔═╡ abb68162-54a1-4a41-9b91-c622d48f773a
map(bitstring, 'a':'z')

# ╔═╡ 8ab0c6dd-0037-4ba5-8a92-a52cdc24630d
md"""
You may notice that bits in lowercase letters change only within a binary mask `01011111000000000000000000000000`. Now let's take a look at the uppercase letters:
"""

# ╔═╡ 648ac7d7-abc9-4b08-9bc3-ddb32be96722
map(bitstring, 'A':'Z')

# ╔═╡ b9769222-1e39-4738-9fa9-001c7802a9de
md"""
Interesting. They look almost identical with just a single difference - they don't have a 3 most significant bit set. So, it seems like we can turn a lowercase letter into uppercase by unsetting 3rd most significant bit. To do this, we can use the one more observation:
"""

# ╔═╡ 354d2f1c-5ce6-48a7-81aa-8ed5787717cc
bitstring('_')

# ╔═╡ 6fd154d2-66c4-4fd9-b221-8ad6b5a0115d
md"""
Which covers all lowercase letter bits but has 0 as the 3rd most significant unset. That's all we need to implement a `to_upper` function:
"""

# ╔═╡ 3c445aec-107a-11ed-34fc-4dabae37652b
to_upper(ch::Char) = Char(Int(ch) & Int('_'))

# ╔═╡ bf204634-80a7-4763-9d25-4267c1befcfe
md"""
And here is how it looks like in C++
"""

# ╔═╡ 91dc037c-d8d9-47fc-8e71-e88c8fac638c
html"""<iframe width="800px" height="600px" src="https://godbolt.org/e#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM6SuADJ4DJgAcj4ARpjEIACcpAAOqAqETgwe3r56KWmOAiFhkSwxcQDMtpj2BQxCBEzEBFk%2BfgF2mA4Z9Y0ERRHRsQm2DU0tOVwjvf0lZSDlAJS2qF7EyOwc5uWhyN5YANQm5W4KBMShwEfYJhoAgje3yAiN%2BywAngD6RB9eSUmxECeLyeC0OAHYrHd9tD9sRMARVgx9k9DmYAGz7MAcD5Yo6Q%2B5ggAiDweQOI%2B1O6C%2BqB%2BfwBZORCFBJghDxhsPhiP2RF%2B/2IgKZeJJRI4S1onAArLw/BwtKRUJw3NZrBSVmtMKjyjxSARNKKlgBrEASjT6TiSXgsCQaU0yuUKji8BQgU262Wi0hwWAwRAoVAsJJ0WLkShoANBuK7QzAKQ2vh0AixZ0QKJ60hRUKNN6cbVhtiCADyDFo2fdpCwLGj4jL%2BDhXQAbphnWXMKpOl5EzneKFE%2BKy7Q8FFiFmPFg02c8FbuB6qAZgAoAGp4TAAdwL/xl2v4ghEYnYUhkgkUKnUZd0kwMRhQyss%2BkHztgzDYIBiDCeleIBtIjbi3EkWgWJZUCSWpmwAWkpI5CVMSxrDMJh9jAgtykQqhTnOBwwPrBoonoMCUh7JMnWqTpahcBh3E8Vo9GCUIBlKIZJjydIBHGPwmNSFiGBmQY4kmDougEHoxioiYqhqbpRj6OjZkYqYROydj5Ok4peIkJYFDVdY9DOTANh4MVJWlNMHVUAAONEwLRSRkUvYB9ikAA6DRnP2CAlVg299lwQgSE1SZ9g8cN6HJLYuAWXg3QAw1jVNPsLVIK0TVIO1eAdJ0XR1PUli9X0w0DEKQ0Bf0CqGKMjFjU0aFoRNiGTVMywzZhiFLXN/XzAgixLNMKyrDY5VrUi8EbZs5VbdtO2nchBGqNMByHEcMH6yLzinAy%2BDnRdlzXDcu0PYRRHEA9t3kJQ1DTXQAjs69PJseaHwgICQIycDIPKaCbwseDEOQ4iJOcCBXDYmiKJ4hi%2BOSTjamBjj8gyMG5n4kjBLqKSYfEoahKkhG5NOXp0bxpocb4jStP3HU4X0j0%2BylFKTM4czLOs2zowcyRnNc9zPtIbz8CIUKzC1HmgtKgXyjMCKsvdQDSAQTAmCwOJHrNDgEqS216cdWxMqi/VSCNZK%2B3KYyy3SqXopVswTftThIuypYfzSZxJCAA%3D"></iframe>"""

# ╔═╡ 1b5b3f50-2479-4662-add9-e4f63625fe77
md"""
So the implementation boilds down to `and     al, 95` which can hardly be more efficient. But does it matter in practice?
"""

# ╔═╡ 46358983-a701-405f-9f75-6d021629694a
html"""
<iframe width="800px" height="600px" src="https://quick-bench.com/q/DzPVsl3mGWvXugCSezMhR2o2G3w"></iframe>
"""

# ╔═╡ 4c125ae7-5f25-4f0a-a68b-ba2a7deedd1b
md"""
8X speedup! Not bad for such a simple implementation. As always, it's unlikely that `toupper` is a bottleneck for your application or service, but this serves as a reminder that sometimes looking under the hood may reveal interesting patterns leading to all sorts of unexpected insights.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╟─73267881-56f3-4a04-a96e-d2f62bda61e4
# ╠═abb68162-54a1-4a41-9b91-c622d48f773a
# ╟─8ab0c6dd-0037-4ba5-8a92-a52cdc24630d
# ╠═648ac7d7-abc9-4b08-9bc3-ddb32be96722
# ╟─b9769222-1e39-4738-9fa9-001c7802a9de
# ╠═354d2f1c-5ce6-48a7-81aa-8ed5787717cc
# ╟─6fd154d2-66c4-4fd9-b221-8ad6b5a0115d
# ╠═3c445aec-107a-11ed-34fc-4dabae37652b
# ╟─bf204634-80a7-4763-9d25-4267c1befcfe
# ╟─91dc037c-d8d9-47fc-8e71-e88c8fac638c
# ╟─1b5b3f50-2479-4662-add9-e4f63625fe77
# ╟─46358983-a701-405f-9f75-6d021629694a
# ╟─4c125ae7-5f25-4f0a-a68b-ba2a7deedd1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
