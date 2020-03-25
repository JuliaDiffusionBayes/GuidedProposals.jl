using GuidedProposals, OrdinaryDiffEq
using Test

@testset "GuidedProposals.jl" begin
    # Write your own tests here.

    struct LVAux
        α::Float64
        β::Float64
        γ::Float64
        δ::Float64
        σ1::Float64
        σ2::Float64
    end

    function GuidedProposals.B!(save_to, t, P::LVAux)
        save_to[1,1] = 0.0
        save_to[1,2] = -P.β * P.γ/P.δ
        save_to[2,1] = P.α * P.δ/P.β
        save_to[2,2] = 0.0
    end

    function GuidedProposals.β!(save_to, t, P::LVAux)
        save_to[1] = P.δ*P.α/P.δ
        save_to[2] = -P.α*P.γ/P.β
    end

    function GuidedProposals.σ!(save_to, t, P::LVAux)
        save_to[1,1] = P.σ1
        save_to[1,2] = 0.0
        save_to[2,1] = 0.0
        save_to[2,2] = P.σ2
    end

    function GuidedProposals.a!(save_to, t, P::LVAux)
        save_to[1,1] = P.σ1^2
        save_to[1,2] = 0.0
        save_to[2,1] = 0.0
        save_to[2,2] = P.σ2^2
    end

    GuidedProposals.dimension(::LVAux) = (
        process = 2,
        wiener = 2,
    )



    params = (
        tt = 0.0:0.01:1.0,
        P_target = nothing,
        P_aux = LVAux(2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2),
        obs = (
            obs = [1.0, 2.0],
            Σ = [1.0 0.0; 0.0 1.0],
            Λ = [1.0 0.0; 0.0 1.0],
            μ = [0.0, 0.0],
            L = [1.0 0.0; 0.0 1.0],
        ),
        solver_choice=(
            solver=Tsit5(),
            ode_type=:HFc,
            convert_to_HFc=false,
            inplace=true,
            save_as_type=nothing,
            ode_data_type=nothing,
        ),
        next_guiding_term=nothing,
    )

    gp = GuidProp(params...)


    H(gp, 3)
    F(gp, 10)
    c(gp, 15)
end
