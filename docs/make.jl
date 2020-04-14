using Documenter, GuidedProposals

makedocs(;
    modules=[GuidedProposals],
    format=Documenter.HTML(
        mathengine = Documenter.MathJax(
            Dict(
                :TeX => Dict(
                    :equationNumbers => Dict(
                        :autoNumber => "AMS"
                    ),
                    :Macros => Dict(
                        :dd => "{\\textrm d}",
                        :RR => "\\mathbb{R}",
                        :wt => ["\\widetilde{#1}", 1]
                    ),
                )
            )
        )
    ),
    pages=[
        "Home" => "index.md",
        "Overview" => Any[
            "Guided proposals" => joinpath("overview", "guid_prop.md"),
            "Multiple observations" => joinpath("overview", "multiple_obs.md"),
            "BFFG algorithm" => joinpath("overview", "bffg.md"),
            "Convenience functions" => joinpath("overview", "convenience.md"),
        ],
        "Advance use" => Any[
            "ODE systems" => joinpath("advanced_use", "ode_type.md"),
            "Additional options" => joinpath("advanced_use", "additional_options.md"),
        ],
        "Index" => "module_index.md",
    ],
    repo="https://github.com/JuliaDiffusionBayes/GuidedProposals.jl/blob/{commit}{path}#L{line}",
    sitename="GuidedProposals.jl",
    authors="Sebastiano Grazzi, Frank van der Meulen, Marcin Mider, Moritz Schauer",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaDiffusionBayes/GuidedProposals.jl",
)
