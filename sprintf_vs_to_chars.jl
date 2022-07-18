### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 15a9e260-0645-11ed-3cca-0b8398e68aa5
md"""
# sprintf vs std::to_chars.
## Or the cost of hidden dependencies.

Most of us have probably heard at least some horror stories about iostream and/or stringstream. As such, when faced with a task to convert some numeric value to its string representation, many instinctively reach out to sprintf/snprintf. It’s C, after all, so what can possibly be faster than it, right?

Well, let’s play with code below
"""

# ╔═╡ 6002b53d-1911-4ae3-be6f-b2f6e6617121
html"""
<iframe width="800px" height="600px" src="https://compiler-explorer.com/e#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:46,endLineNumber:13,positionColumn:46,positionLineNumber:13,selectionStartColumn:46,selectionStartLineNumber:13,startColumn:46,startLineNumber:13),source:'%23include+%3Ccharconv%3E%0A%23include+%3Carray%3E%0A%23include+%3Ccstdio%3E%0A%23include+%3Ciostream%3E%0A%23include+%3Cmemory%3E%0A%23include+%3Cstring_view%3E%0A%0Aconstexpr+size_t+N+%3D+10%3B%0A%0Astd::string_view+convertToChars(char+buf%5BN%5D,+double+d)+%7B%0A++++auto+%5Bptr,+errc%5D+%3D+std::to_chars(buf,+buf+%2B+N,+d)%3B%0A++++//+TODO:+handle+errc%0A++++//+if+(errc+!!%3D+std::errc())+throw+%22oops%22%3B%0A++++return+std::string_view(buf,+ptr+-+buf)%3B%0A%7D%0A%0Achar*+convertSprintf(char+buf%5BN%5D,+double+d)+%7B%0A++++std::snprintf(buf,+N,+%22%25f%22,+d)%3B%0A++++return+buf%3B%0A%7D%0A%0Aint+main()+%7B%0A++++char+buf%5B10%5D%3B%0A++++std::cout+%3C%3C+convertToChars(buf,+3.141592)+%3C%3C+std::endl%3B%0A++++std::cout+%3C%3C+convertSprintf(buf,+3.141592)+%3C%3C+std::endl%3B%0A%7D'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((g:!((h:compiler,i:(compiler:gsnapshot,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3+-Wall+-Wpedantic+-std%3Dc%2B%2B17',selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1,tree:'1'),l:'5',n:'0',o:'x86-64+gcc+(trunk)+(C%2B%2B,+Editor+%231,+Compiler+%231)',t:'0')),k:50,l:'4',m:50,n:'0',o:'',s:0,t:'0'),(g:!((h:executor,i:(argsPanelShown:'1',compilationPanelShown:'0',compiler:g121,compilerOutShown:'0',execArgs:'',execStdin:'',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3+-Wall+-Wpedantic+-std%3Dc%2B%2B17',source:1,stdinPanelShown:'1',tree:'1',wrap:'1'),l:'5',n:'0',o:'Executor+x86-64+gcc+12.1+(C%2B%2B,+Editor+%231)',t:'0')),header:(),l:'4',m:50,n:'0',o:'',s:0,t:'0')),k:50,l:'3',n:'0',o:'',t:'0')),l:'2',n:'0',o:'',t:'0')),version:4"></iframe>
"""

# ╔═╡ e95a8a43-090b-4770-90f6-2dda73f93328
md"""
There are serveal things to note:
- `std::to_chars` is safe by default - it makes sure that resulting string does not exceed provided buffer and returns signals error conditions, if any
- there are a few more assembly instructions involved in setting up a call to `sprintf` but nothing suspicious.

But how do these functions perform in benchmark?
"""

# ╔═╡ 74fe7549-f350-45aa-9f85-aca2aab1818d
html"""
<iframe width="800px" height="600px" src="https://quick-bench.com/q/PYL_PnLT5Brkzl08kNNXyv5kwXw"></iframe>
"""

# ╔═╡ d622e5ad-4a9c-4273-990c-3c67f98ea034
md"""
Ouch, `sprintf` is 2.8X slower! That's a huge difference for such a small amount of work. So what's the deal?

Unfortunately, I don't have a setup for profiling these functions, so I can only speculate, but there are at least 2 suspects:
- `sprintf` is locale aware and, for example, [uses](https://github.com/lattera/glibc/blob/master/stdio-common/vfprintf.c#L1412) `LC_NUMERIC` to resolve thousands separator
- `sprintf` has a lot of complexity and shares its implementation with a large number of functions, including `fprintf` which works with file descriptors instead of string buffers directly. While such unification avoids significant amount of duplication across many formatting functions, it also adds overhead.

### Conclusion?
There is really no reason to use `sprintf` to format numeric values - `std::to_chars` is simple to use, safe and fast.
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
# ╟─15a9e260-0645-11ed-3cca-0b8398e68aa5
# ╟─6002b53d-1911-4ae3-be6f-b2f6e6617121
# ╟─e95a8a43-090b-4770-90f6-2dda73f93328
# ╟─74fe7549-f350-45aa-9f85-aca2aab1818d
# ╟─d622e5ad-4a9c-4273-990c-3c67f98ea034
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
