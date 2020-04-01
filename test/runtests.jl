using GuidedProposals, DiffusionDefinition
using OrdinaryDiffEq, StaticArrays
using ForwardDiff, Random
using Test
const DD = DiffusionDefinition
const GP = GuidedProposals

@testset "GuidedProposals.jl" begin
    # Write your own tests here.
    @load_diffusion LotkaVolterraAux

    function DD.B!(save_to, t, P::LotkaVolterraAux)
        save_to[1,1] = 0.0
        save_to[1,2] = -P.β * P.γ/P.δ
        save_to[2,1] = P.α * P.δ/P.β
        save_to[2,2] = 0.0
    end

    function DD.β!(save_to, t, P::LotkaVolterraAux)
        save_to[1] = P.γ*P.α/P.δ
        save_to[2] = -P.α*P.γ/P.β
    end

    function DD.σ!(save_to, t, P::LotkaVolterraAux)
        save_to[1,1] = P.σ1
        save_to[1,2] = 0.0
        save_to[2,1] = 0.0
        save_to[2,2] = P.σ2
    end

    function DD.a!(save_to, t, P::LotkaVolterraAux)
        save_to[1,1] = P.σ1^2
        save_to[1,2] = 0.0
        save_to[2,1] = 0.0
        save_to[2,2] = P.σ2^2
    end

    Base.eltype(::LotkaVolterraAux{T}) where T = T

    P = LotkaVolterraAux(
        2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2, 0.0,0.0, nothing, nothing
    )

    params_intv1 = (
        tt = 0.0:0.01:1.0,
        P_target = P,
        P_aux = P,
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
            mode=:inplace,
            gradients=false,
            eltype=Float64,
        )
    )

    params_intv2 = (
        tt = 1.0:0.01:2.0,
        P_target = P,
        P_aux = P,
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
            mode=:inplace,
            gradients=false,
            eltype=Float64,
        )
    )

    gp2 = GuidProp(params_intv2..., nothing)
    gp1 = GuidProp(params_intv1..., gp2)

    recompute_guiding_term(gp2, nothing)
    recompute_guiding_term(gp1, gp2)

    H(gp2, 3)
    F(gp2, 10)
    c(gp2, 15)

    H(gp1, 3)
    F(gp1, 10)
    c(gp1, 15)

    params_intv1 = (
        tt = 0.0:0.01:1.0,
        P_target = P,
        P_aux = P,
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
            mode=:outofplace,
            gradients=false,
            eltype=Float64,
        )
    )

    params_intv2 = (
        tt = 1.0:0.01:2.0,
        P_target = P,
        P_aux = P,
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
            mode=:outofplace,
            gradients=false,
            eltype=Float64,
        )
    )

    gp2 = GuidProp(params_intv2..., nothing)
    gp1 = GuidProp(params_intv1..., gp2)

    recompute_guiding_term(gp2, nothing)
    recompute_guiding_term(gp1, gp2)

    H(gp2, 3)
    F(gp2, 10)
    c(gp2, 15)

    H(gp1, 3)
    F(gp1, 10)
    c(gp1, 15)

    N = 2
    tt = reverse(gp2.guiding_term_solver.saved_values.t)
    WW = trajectory(tt, SVector{N,Float64})
    wr = Wiener()
    rand!(WW, wr)

    XX = trajectory(tt, SVector{N,Float64})
    x0 = @SVector [1.0, 2.0]
    DD.solve!(XX, WW, gp2, x0)

    #[TODO move to DiffusionDefinition.jl]
    #@inline DD.constdiff(P::LotkaVolterraAux) = true

    loglikelihood(XX, gp2)
    loglikelihood_obs(gp2, x0)

    # in DiffusionDefinition it asserts Float64 eltype, change it there
    DD.β(t, P::LotkaVolterraAux) = @SVector [P.γ/P.δ*P.α, -P.α/P.β*P.γ]

    function foo(θ)
        params_intv2 = (
            tt = 1.0:0.01:2.0,
            P_target = LotkaVolterraAux(θ..., 0.0,0.0, nothing, nothing),
            P_aux = LotkaVolterraAux(θ..., 0.0,0.0, nothing, nothing),
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
                mode=:outofplace,
                gradients=true,
                eltype=Float64,
            )
        )

        gp2 = GuidProp(params_intv2..., nothing)
        recompute_guiding_term(gp2, nothing)

        sum(gp2.guiding_term_solver.HFc0)
    end

    grad = ForwardDiff.gradient(foo, [2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2])
end
