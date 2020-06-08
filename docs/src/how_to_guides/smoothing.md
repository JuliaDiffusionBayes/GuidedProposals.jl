# [How to do smoothing of diffusion trajectories?](@id how_to_smoothing)
***
**Smoothing** is a process of reconstructing the unobserved parts of the path, based on the recorded observations.

```julia
# Perform smoothing for the data in the `recording`, using Guided Proposals with
# the auxiliary law `AuxLaw`.
function simple_smoothing(AuxLaw, recording, dt; ρ=0.5, num_steps=10^4)
    # -------------------------------------------------------------------------#
    #                          Initializations                                 #
    # -------------------------------------------------------------------------#
    # time-grids for the forward-simulation of trajectories                    #
    tts = OBS.setup_time_grids(recording, dt)                                  #
    # memory parameters for the preconditioned Crank-Nicolson scheme           #
    ρρ = [ρ for _ in tts]                                                      #
    # laws of guided proposals                                                 #
    PP = build_guid_prop(AuxLaw, recording, tts)                               #
                                                                               #
    # starting point                                                           #
    # NOTE `rand` for `KnownStartingPt` simply returns the starting position   #
    y1 = rand(recording.x0_prior)                                              #
    # initialize the `accepted` trajectory                                     #
    XX, WW, Wnr = rand(PP, y1)                                                 #
    # initialize the containers for the `proposal` trajectory                  #
    XX°, WW° = trajectory(PP)                                                  #
                                                                               #
    ll = loglikhd(PP, XX)                                                      #
    paths = []                                                                 #
    num_accpt = 0                                                              #
    # -------------------------------------------------------------------------#

    # MCMC
    for i in 1:num_steps
        # impute a path
        _, ll° = rand!(PP, XX°, WW°, WW, ρρ, Val(:ll), y1; Wnr=Wnr)

        # Metropolis–Hastings accept/reject step
        if rand() < exp(ll°-ll)
            XX, WW, XX°, WW° = XX°, WW°, XX, WW
            ll = ll°
            num_accpt += 1
        end

        # progress message
        if i % 100 == 0
            println("$i. ll=$ll, acceptance rate: $(num_accpt/100)")
            num_accpt = 0
        end

        # save intermediate path for plotting
        i % 400 == 0 && append!(paths, [deepcopy(XX)])
    end
    paths
end
```

### Example
For instance, for a partially observed FitzHugh–Nagumo model

```julia
recording = ...
paths = simple_smoothing(
    FitzHughNagumoAux, recording, 0.001; ρ=0.96, num_steps=10^4
)
```
![paths](../assets/how_to/smoothing/paths.png)

It takes about 6sec on my laptop.
