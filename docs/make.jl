using BigRiverMetabolomics
using Documenter
using DocumenterVitepress

makedocs(;
    modules = [BigRiverMetabolomics],
    authors = "Abhirath Anand <74202102+theabhirath@users.noreply.github.com> and contributors",
    sitename = "BigRiverMetabolomics.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/senresearch/BigRiverMetabolomics.jl",
    ),
    pages = ["Home" => "index.md"]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/senresearch/BigRiverMetabolomics.jl",
    target = "build",
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true
)
