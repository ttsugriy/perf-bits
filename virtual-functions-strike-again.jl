### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ cf8abc66-0002-11ed-083d-e57e6b3dafc5
md"""
# Virtual functions strike again.
## Or too much dynamism is a thing.

Much has already been written about virtual dispatch overhead. They take up more space, add memory indirection, prevent most compiler optimizations and as such are claerly not the best choice when it comes to performance.

But as if the list above is not bad enough, there is another less known fact about them - since virtual table pointers can be changed, compilers, to ensure that every virtual call is dispatched to the right function have to resolve them on each invocation.

We can clearly see this in the example below
"""

# ╔═╡ 1fe064ff-52c4-4d58-945d-1fc4741f2ff5
html"""
<iframe width="100%" height="400px" src="https://godbolt.org/e#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:29,endLineNumber:14,positionColumn:29,positionLineNumber:14,selectionStartColumn:29,selectionStartLineNumber:14,startColumn:29,startLineNumber:14),source:'class+Parent+%7B%0Apublic:%0A++++virtual+int+foo()+const+%3D+0%3B%0A%7D%3B%0A%0Aclass+Child+:+Parent+%7B%0Apublic:%0A++++int+foo()+const+override+%7B%0A++++++++return+3%3B%0A++++%7D%0A%7D%3B%0A%0Aint+playground(const+Parent%26+p)+%7B%0A++++return+p.foo()+%2B+p.foo()%3B%0A%7D'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',m:100,n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:clang1400,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:2,lang:c%2B%2B,libs:!(),options:'-std%3Dc%2B%2B2a+-O3',selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1,tree:'1'),l:'5',n:'0',o:'x86-64+clang+14.0.0+(C%2B%2B,+Editor+%231,+Compiler+%232)',t:'0')),header:(),k:50,l:'4',m:100,n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4"></iframe>
"""

# ╔═╡ 79828898-67a5-494b-bc39-9ef86f5ec676
md"""
But how often do you override your virtual table pointers? Probably not that often. If only there was a way to let compiler know this...

Fortunately, there is way and it boils down to a simple copmiler flag `-fstrict-vtable-pointers`. It's a promise to a compiler that the virtual vptr remains invariant during object's lifetime. This means that compiler is allowed to reuse vptr instead of reloading it on each invocation.

The difference is exactly what we'd expect
"""

# ╔═╡ 941d13d6-db79-49ac-a02a-6d795d7bab0c
html"""
<iframe width="800px" height="600px" src="https://godbolt.org/e#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIMwCcpK4AMngMmAByPgBGmMQS0gAOqAqETgwe3r7%2BQSlpjgJhEdEscQlStpj2hQxCBEzEBFk%2BfoFVNRn1jQTFUbHxibYNTS057QojveH9ZYNSAJS2qF7EyOwcyAYKCgDUyo2MBLsmAOxWGgCCSV4xtHjIICZXu6%2B7AG54TV5iu%2BHH/FQEAWuzQDEmJwAzAARXYaEyQi6XM7QhFI56XLZMHa7NwIOjoXYgfaHQQnc4Ym53B5PF5vf67QHA0ECCGod7xYh4LDkpFvfm7YiYAirBi7SFojH8lEYmWI2VXBlJAwAT2AxBWDHQEDBEIOQsE5gAbLskiCzny3kKRcQxUkAHRM82WU2O1BAhaSq4ojhLWicACsvD8HC0pFQnDc1msuwUKzWmBOZkhPFIBE0vqWAGsQAGNPpOJJeCwJBp8yGwxGOLwFCB8%2BnQ77SHBYDBEChUCwknR4uRKGguz2ElijFIy3w6AR4rWIDEM6QYuFGirOKmB2xBAB5Bi0FeN0hYFiGYDiff4IUOPAc2v7zCqTDILxT1e8f7Vef3GLEZceLDzghciW3BNlQBjAAoABqeCYAA7puSSMC%2BMiCCIYjsJU/CCIoKjqPuuhcPox4oNGlj6HgMS1rAzBsCAcQMMgCBHsQWakByCTcJIWgLEsqBJLUN4ALSTOgCLQqYljWGYTC7AJm6QjJVCTFyDgCe8DR3JgAkpG%2BxAKDW1QPrULhamMfgEaEMylOUej5OkAimTZqR2QwfRWfMHSGV0UwOQRdieQI3RNK5AwVMMPQ%2BWFQWWSFEhLHGqzrHoAGYBsPB%2BoGwbzlWqgABxGgJRqSKCBhGLsUj2hoFW7BAUYSaRuy4IQJBJimpC4p23b0MQLVcAsvANlx2a5vm/ocEWpAlnmpAVrwVY1nWaYZksLbtgOnW9hQOodUOKAlcAY75jQtBTrplBzvui7MMQe5rp2G4ENuu7zoex6nmG56ede853g%2BT6pa%2Bgjvvun7ftdv4bGGAF4EBaV8GBkHQXBCEhqmmHCKI4gYbI2FqPOuhmIRRjEXVNifpRwLhnxGSCcJoniVYlhSTJcn6Z0zgQK4EUWSUMUEbZtQRfzGTBXMoV%2BZeAXeZ4rR6OLtSBdMPOi7LUvZGZkWK7M1m9csCXoWmQr/elHBBtNWWcLl%2BWFcVx5lZIFVVTVJE2A1%2BBEN15ite1g5dS1Zh9YtjbcaQCCYEwWAJBTo3jZN5bm9WtgLQNmakDmU2jZCmX7nNgeDQWHBmFnlacP1S1LGxaTOJIQA%3D%3D"></iframe>
"""

# ╔═╡ 00e80dda-6c3e-4cc8-9e49-eddee7bd02c1
md"""
Runtime performance impact of this optimization depends on whether virtual calls are on a hot path and your branch predictor, so as always, don't forget to benchmark before applying it.
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
# ╟─cf8abc66-0002-11ed-083d-e57e6b3dafc5
# ╟─1fe064ff-52c4-4d58-945d-1fc4741f2ff5
# ╟─79828898-67a5-494b-bc39-9ef86f5ec676
# ╟─941d13d6-db79-49ac-a02a-6d795d7bab0c
# ╟─00e80dda-6c3e-4cc8-9e49-eddee7bd02c1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
