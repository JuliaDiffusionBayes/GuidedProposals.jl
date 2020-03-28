using Documenter, GuidedProposals

makedocs(;
    modules=[GuidedProposals],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaDiffusionBayes/GuidedProposals.jl/blob/{commit}{path}#L{line}",
    sitename="GuidedProposals.jl",
    authors="Marcin Mider",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaDiffusionBayes/GuidedProposals.jl",
)
