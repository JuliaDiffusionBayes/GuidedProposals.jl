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
                    :packages => Dict(
                        "[+]" => ["color"]
                    ),
                ),
                :loader => Dict(
                    :load => ["[tex]/color"],
                ),
            ),
        ),
        collapselevel = 1,
    ),
    pages=[
        "Home" => "index.md",
        "Get started" => joinpath("get_started", "overview.md"),
        "User manual" => Any[
            "Guided proposals" => joinpath("manual", "guid_prop.md"),
            "Multiple observations" => joinpath("manual", "multiple_obs.md"),
            "Log-likelihoods" => joinpath("manual", "log_likelihood.md"),
            "Reparameterizations" => joinpath("manual", "reparameterizations.md"),
            "BFFG algorithm" => joinpath("manual", "bffg.md"),
            "Computing path functionals" => joinpath("manual", "path_functionals.md"),
            "(TODO) ODE systems" => joinpath("manual", "ode_type.md"),
        ],
        "How to..." => Any[
            "(TODO) Sample diffusion bridges" => joinpath("how_to_guides", "sample_bridges.md"),
            "(TODO) Sample diffusions in first-passage time setting" => joinpath("how_to_guides", "first_passage_time.md"),
            "(TODO) Do smoothing" => joinpath("how_to_guides", "smoothing.md"),
            "(TODO) Do parameter inference" => joinpath("how_to_guides", "parameter_inference.md")
        ],
        "Tutorials" => Any[
            "(TODO) Understanding in-place vs out-of-place" => joinpath("tutorials", "in_place_vs_out_of_place.md"),
            "(TODO) Smoothing/Imputation" => joinpath("tutorials", "smoothing.md"),
            "Parameter inference" => joinpath("tutorials", "parameter_inference.md"),
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
