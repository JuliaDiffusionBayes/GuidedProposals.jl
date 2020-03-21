using Documenter, GuidedProposals

makedocs(;
    modules=[GuidedProposals],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/mmider/GuidedProposals.jl/blob/{commit}{path}#L{line}",
    sitename="GuidedProposals.jl",
    authors="Marcin Mider",
    assets=String[],
)

deploydocs(;
    repo="github.com/mmider/GuidedProposals.jl",
)
