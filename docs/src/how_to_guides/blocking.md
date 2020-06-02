# How to do blocking?
****
Blocking is a technique that modifies smoothing algorithms and facilitates more efficient exploration of the path space. A path is updated in blocks instead of being imputed in full. Blocking and the preconditioned Crank-Nicolson scheme both aim to achieve the same goal, but the two approaches differ in execution.

We will explain how to do blocking on the example of a smoothing problem from the previous how-to-guide: [smoothing of the partially observed FitzHugh–Nagumo trajectories](@ref how_to_smoothing).

```julia
τ(t₀,T) = (t) ->  t₀ + (t-t₀) * (2-(t-t₀)/(T-t₀))
τ(tt) = τ(tt[1], tt[end]).(tt)

# and define a function that does the inference
function simple_smoothing_with_blocking(AuxLaw, recording, dt; ρ=0.5, num_steps=10^4)
    # initializations
    tts = OBS.setup_time_grids(recording, dt, τ) # passing additionally time-transformation
    ρρ = [ρ for _ in tts]
    PP = build_guid_prop(AuxLaw, recording, tts; artifical_noise=1e-6)

    y1 = rand(recording.x0_prior) # just returns the starting point
    XX, WW, Wnr = rand(PP, y1)
    XX°, WW° = trajectory(PP)

    ll = loglikhd(PP, XX)
    paths = []

    #--------------------------------------------------------------------#
    #                       Blocking setup                               #
    #--------------------------------------------------------------------#
    # let's do some very simple blocking based on three points           #
    num_intv = length(PP)                                                #
    one_quarter_pt = div(num_intv, 4)                                    #
    one_half_pt = div(num_intv, 2)                                       #
    three_quarter_pt = one_half_pt + one_quarter_pt                      #
    block_set_1_builder(x) = [                                           #
        view(x, 1:one_half_pt),                                          #
        view(x, (one_half_pt+1):length(x))                               #
    ]                                                                    #
    block_set_2_builder(x) = [                                           #
        view(x, 1:one_quarter_pt),                                       #
        view(x, (one_quarter_pt+1):three_quarter_pt),                    #
        view(x, (three_quarter_pt+1):length(x))                          #
    ]                                                                    #
    make_block_set(f) = (                                                #
        PP = f(PP),                                                      #
        XX = f(XX),                                                      #
        XX° = f(XX°),                                                    #
        WW = f(WW),                                                      #
        WW° = f(WW°),                                                    #
        ρρ = f(ρρ),                                                      #
    )                                                                    #
    B1 = make_block_set(block_set_1_builder)                             #
    B2 = make_block_set(block_set_2_builder)                             #
    imp_a_r = [[0,0], [0,0,0]]                                           #
    lls = [[0.0, 0.0], [0.0, 0.0, 0.0]]                                  #
    #--------------------------------------------------------------------#

    # MCMC
    for i in 1:num_steps
        #  -----------------------
        #  | imputation on SET 1 |
        #  -----------------------
        # set an auxiliary point
        set_aux_obs!(
            B1.PP[1][end],
            B1.XX[2][1].x[1] # set 1, second block, first XX, first obs
        )
        recompute_guiding_term!(B1.PP[1]; blocking=BlockingMode())
        recompute_guiding_term!(B1.PP[2])

        # find a Wiener path W reconstructing the trajectory X
        for j in 1:2
            for k in 1:length(B1.XX[j])
                DD.invsolve!(B1.XX[j][k], B1.WW[j][k], B1.PP[j][k])
            end
        end

        y = y1
        # impute the path
        for j in 1:2 # there are two blocks in the first set

            # sample a path on a given block
            _, ll° = rand!(B1.PP[j], B1.XX°[j], B1.WW°[j], B1.WW[j], B1.ρρ[j], Val(:ll), y; Wnr=Wnr)

            # compute log-likelihood on this interval for the accepted path
            ll = loglikhd(B1.PP[j], B1.XX[j])

            lls[1][j] = ll
            if rand() < exp(ll°-ll)
                B1.XX[j], B1.WW[j], B1.XX°[j], B1.WW°[j] = B1.XX°[j], B1.WW°[j], B1.XX[j], B1.WW[j]
                imp_a_r[1][j] += 1
                lls[1][j] = ll°
            end
            y = B1.XX[j][end].x[end]
        end

        #  -----------------------
        #  | imputation on SET 2 |
        #  -----------------------
        # set auxiliary points
        for j in 1:2
            set_aux_obs!(
                B2.PP[j][end],
                B2.XX[j][end].x[end] # set 2, jth block, last XX, last obs
            )
            recompute_guiding_term!(B2.PP[j]; blocking=BlockingMode())
        end
        # B2.PP[3] does not need to be recomputed, but of course, can be for good measure

        # find a Wiener path W reconstructing the trajectory X
        for j in 1:3
            for k in 1:length(B2.XX[j])
                DD.invsolve!(B2.XX[j][k], B2.WW[j][k], B2.PP[j][k])
            end
        end

        y = y1
        # impute the path
        for j in 1:3 # there are three blocks in the second set

            # sample a path on a given block
            _, ll° = rand!(B2.PP[j], B2.XX°[j], B2.WW°[j], B2.WW[j], B2.ρρ[j], Val(:ll), y; Wnr=Wnr)

            # compute log-likelihood on this interval for the accepted path
            ll = loglikhd(B2.PP[j], B2.XX[j])

            lls[2][j] = ll
            if rand() < exp(ll°-ll)
                B2.XX[j], B2.WW[j], B2.XX°[j], B2.WW°[j] = B2.XX°[j], B2.WW°[j], B2.XX[j], B2.WW[j]
                imp_a_r[2][j] += 1
                lls[2][j] = ll°
            end
            y = B2.XX[j][end].x[end]
        end

        # progress message
        if i % 100 == 0
            println(
                "$i. ",
                "ll₁₁=$(lls[1][1]) (ar₁₁=$(imp_a_r[1][1]/100)), ",
                "ll₁₂=$(lls[1][2]) (ar₁₂=$(imp_a_r[1][2]/100));  ",
                "ll₂₁=$(lls[2][1]) (ar₂₁=$(imp_a_r[2][1]/100)), ",
                "ll₂₂=$(lls[2][2]) (ar₂₂=$(imp_a_r[2][2]/100)), ",
                "ll₂₃=$(lls[2][3]) (ar₂₃=$(imp_a_r[2][3]/100))"
            )
            imp_a_r[1] .= 0
            imp_a_r[2] .= 0
        end

        # save intermediate path for plotting
        i % 400 == 0 && append!(paths, [deepcopy(XX)])
    end
    paths
end

@inline DD.nonhypo(x, P::FitzHughNagumo) = x[SVector{1,Int64}(2)]
@inline DD.nonhypo_σ(t::Float64, x, P::FitzHughNagumo) = SMatrix{1,1,Float64}(P.σ)

# and do the inference
paths = simple_smoothing_with_blocking(
    FitzHughNagumoAux, recording, 0.001; ρ=0.9, num_steps=10^4
)
```
