using GuidedProposals, OrdinaryDiffEq, StaticArrays
using ForwardDiff # remember to remove ForwardDiff from dependencies
using Test

@testset "GuidedProposals.jl" begin
    # Write your own tests here.

    struct LVAux{T}
        α::T
        β::T
        γ::T
        δ::T
        σ1::T
        σ2::T
        LVAux(α::T,β,γ,δ,σ1,σ2) where T = new{T}(α,β,γ,δ,σ1,σ2)
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



    params_intv1 = (
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
        )
    )

    params_intv2 = (
        tt = 1.0:0.01:2.0,
        P_target = nothing,
        P_aux = LVAux(2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2),
        obs = (
            obs = [2.0, 3.0],
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
        )
    )

    gp2 = GuidProp(params_intv2..., nothing)
    gp1 = GuidProp(params_intv1..., gp2)

    H(gp2, 3)
    F(gp2, 10)
    c(gp2, 15)

    H(gp1, 3)
    F(gp1, 10)
    c(gp1, 15)


    GuidedProposals.B(t, P::LVAux) = @SMatrix [
        0.0   (-P.β * P.γ/P.δ);
        (P.α * P.δ/P.β)  0.0
    ]

    GuidedProposals.β(t, P::LVAux) = @SVector [P.δ*P.α/P.δ, -P.α*P.γ/P.β]

    GuidedProposals.σ(t, P::LVAux) = @SMatrix [
        P.σ1 0.0;
        0.0 P.σ2
    ]

    GuidedProposals.a(t, P::LVAux) = @SMatrix [
        P.σ1^2 0.0;
        0.0 P.σ2^2
    ]

    params_intv1 = (
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
            inplace=false,
            save_as_type=nothing,
            ode_data_type=nothing,
        )
    )

    params_intv2 = (
        tt = 1.0:0.01:2.0,
        P_target = nothing,
        P_aux = LVAux(2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2),
        obs = (
            obs = [2.0, 3.0],
            Σ = [1.0 0.0; 0.0 1.0],
            Λ = [1.0 0.0; 0.0 1.0],
            μ = [0.0, 0.0],
            L = [1.0 0.0; 0.0 1.0],
        ),
        solver_choice=(
            solver=Tsit5(),
            ode_type=:HFc,
            convert_to_HFc=false,
            inplace=false,
            save_as_type=nothing,
            ode_data_type=nothing,
        )
    )

    gp2 = GuidProp(params_intv2..., nothing)
    gp1 = GuidProp(params_intv1..., gp2)

    H(gp2, 3)
    F(gp2, 10)
    c(gp2, 15)

    H(gp1, 3)
    F(gp1, 10)
    c(gp1, 15)

    function foo(θ)
        params_intv2 = (
            tt = 1.0:0.01:2.0,
            P_target = nothing,
            P_aux = LVAux(θ...),
            obs = (
                obs = [2.0, 3.0],
                Σ = [1.0 0.0; 0.0 1.0],
                Λ = [1.0 0.0; 0.0 1.0],
                μ = [0.0, 0.0],
                L = [1.0 0.0; 0.0 1.0],
            ),
            solver_choice=(
                solver=Tsit5(),
                ode_type=:HFc,
                convert_to_HFc=false,
                inplace=false,
                save_as_type=(Matrix{eltype(θ)}, Vector{eltype(θ)}, eltype(θ)),
                ode_data_type=eltype(θ),
            )
        )

        gp2 = GuidProp(params_intv2..., nothing)

        sum(gp2.guiding_term_solver.HFc0)
    end

    grad = ForwardDiff.gradient(foo, [2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2])

end
