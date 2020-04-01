using GuidedProposals, DiffusionDefinition
using OrdinaryDiffEq, StaticArrays
using ForwardDiff, Random
using Test
const DD = DiffusionDefinition
const GP = GuidedProposals

@testset "GuidedProposals.jl" begin
    Random.seed!(10)
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

    function build_obs(::Val{:outofplace}, obs)
        (
            obs = obs,
            Σ = (@SMatrix [1.0 0.0; 0.0 1.0]),
            Λ = (@SMatrix [1.0 0.0; 0.0 1.0]),
            μ = (@SVector [0.0, 0.0]),
            L = (@SMatrix [1.0 0.0; 0.0 1.0]),
        )
    end

    function build_obs(::Val{:inplace}, obs)
        (
            obs = obs,
            Σ = [1.0 0.0; 0.0 1.0],
            Λ = [1.0 0.0; 0.0 1.0],
            μ = [0.0, 0.0],
            L = [1.0 0.0; 0.0 1.0],
        )
    end

    function build_params(mode, obs, gradients; t0=0)
        params_intv1 = (
            tt = (t0+0.0):0.01:(t0+1.0),
            P_target = P,
            P_aux = P,
            obs = build_obs(Val{mode}(), obs),
            solver_choice=(
                solver=Tsit5(),
                ode_type=:HFc,
                convert_to_HFc=false,
                mode=mode,
                gradients=gradients,
                eltype=Float64,
            )
        )
    end

    gp2_inplace = GuidProp(
        build_params(:inplace, [1.0, 2.0], false; t0=0.0)...,
        nothing
    )
    gp1_inplace = GuidProp(
        build_params(:inplace, [2.0, 3.0], false; t0=1.0)...,
        gp2_inplace
    )

    gp1_inplace_recomputed = deepcopy(gp1_inplace)
    gp2_inplace_recomputed = deepcopy(gp2_inplace)

    recompute_guiding_term(gp2_inplace_recomputed, nothing)
    recompute_guiding_term(gp1_inplace_recomputed, gp2_inplace_recomputed)

    gp2_static = GuidProp(
        build_params(:outofplace, (@SVector [1.0, 2.0]), false; t0=0.0)...,
        nothing
    )
    gp1_static = GuidProp(
        build_params(:outofplace, (@SVector [2.0, 3.0]), false; t0=1.0)...,
        gp2_static
    )

    gp1_static_recomputed = deepcopy(gp1_static)
    gp2_static_recomputed = deepcopy(gp2_static)

    recompute_guiding_term(gp2_static, nothing)
    recompute_guiding_term(gp1_static, gp2_static)

    @testset "computation of H,F,c" begin
        @testset "recomputation inplace" for i in 1:101
            @test H(gp1_inplace_recomputed, i) == H(gp1_inplace, i)
            @test F(gp1_inplace_recomputed, i) == F(gp1_inplace, i)
            @test c(gp1_inplace_recomputed, i) == c(gp1_inplace, i)
            @test H(gp2_inplace_recomputed, i) == H(gp2_inplace, i)
            @test F(gp2_inplace_recomputed, i) == F(gp2_inplace, i)
            @test c(gp2_inplace_recomputed, i) == c(gp2_inplace, i)
        end

        @testset "recomputation outofplace" for i in 1:101
            @test H(gp1_static_recomputed, i) == H(gp1_static, i)
            @test F(gp1_static_recomputed, i) == F(gp1_static, i)
            @test c(gp1_static_recomputed, i) == c(gp1_static, i)
            @test H(gp2_static_recomputed, i) == H(gp2_static, i)
            @test F(gp2_static_recomputed, i) == F(gp2_static, i)
            @test c(gp2_static_recomputed, i) == c(gp2_static, i)
        end

        @testset "inplace vs outofplace" for i in 1:101
            @test all(
                [
                    H(gp1_static, i)[k,l] ≈ H(gp1_inplace, i)[k,l]
                    for k in 1:2 for l in 1:2
                ]
            )
            @test all(
                [
                    F(gp1_static, i)[k] ≈ F(gp1_inplace, i)[k]
                    for k in 1:2
                ]
            )
            @test c(gp1_static, i) ≈ c(gp1_inplace, i)

            @test all(
                [
                    H(gp2_static, i)[k,l] ≈ H(gp2_inplace, i)[k,l]
                    for k in 1:2 for l in 1:2
                ]
            )
            @test all(
                [
                    F(gp2_static, i)[k] ≈ F(gp2_inplace, i)[k]
                    for k in 1:2
                ]
            )
            @test c(gp2_static, i) ≈ c(gp2_inplace, i)
        end
    end

    N = 2
    tt = reverse(gp2_static.guiding_term_solver.saved_values.t)
    WW = trajectory(tt, SVector{N,Float64})
    wr = Wiener()
    rand!(WW, wr)

    XX = trajectory(tt, SVector{N,Float64})
    x0 = @SVector [1.0, 2.0]
    DD.solve!(XX, WW, gp2_static, x0)

    #[TODO move to DiffusionDefinition.jl]
    #@inline DD.constdiff(P::LotkaVolterraAux) = true

    loglikelihood(XX, gp2_static)
    loglikelihood_obs(gp2_static, x0)

    # in DiffusionDefinition it asserts Float64 eltype, change it there
    DD.β(t, P::LotkaVolterraAux) = @SVector [P.γ/P.δ*P.α, -P.α/P.β*P.γ]

    function foo(θ)
        params_intv2 = (
            tt = 1.0:0.01:2.0,
            P_target = LotkaVolterraAux(θ..., 0.0,0.0, nothing, nothing),
            P_aux = LotkaVolterraAux(θ..., 0.0,0.0, nothing, nothing),
            obs = (
                obs = (@SVector [2.0, 3.0]),
                Σ = (@SMatrix [1.0 0.0; 0.0 1.0]),
                Λ = (@SMatrix [1.0 0.0; 0.0 1.0]),
                μ = (@SVector [0.0, 0.0]),
                L = (@SMatrix [1.0 0.0; 0.0 1.0]),
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

    θinit = [2.0/3.0, 4.0/3.0, 1.0, 1.0, 0.2, 0.2]
    grad = ForwardDiff.gradient(foo, θinit)

    out_at_θ = foo(θinit)
    ϵ = 0.01
    out_mod = [
        foo(θinit .+ [ϵ, 0.0, 0.0, 0.0, 0.0, 0.0]),
        foo(θinit .+ [0.0, ϵ, 0.0, 0.0, 0.0, 0.0]),
        foo(θinit .+ [0.0, 0.0, ϵ, 0.0, 0.0, 0.0]),
        foo(θinit .+ [0.0, 0.0, 0.0, ϵ, 0.0, 0.0]),
        foo(θinit .+ [0.0, 0.0, 0.0, 0.0, ϵ, 0.0]),
        foo(θinit .+ [0.0, 0.0, 0.0, 0.0, 0.0, ϵ]),
    ]
    @testset "automatic differentiation" for i in 1:6
        @test abs( (out_mod[i] - out_at_θ)/ϵ - grad[i] ) < 0.1
    end
end
