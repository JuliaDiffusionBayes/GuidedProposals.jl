# Reparameterizations of `GuidProp`
****************
In an MCMC setting we may wish to change the values of some parameters of `GuidProp`. `GuidProp` is defined as an immutable struct; however, the fields containing parameter values are mutable, and thus, changing parameters may be done in-place. Changing them directly by accessing relevant places that store parameter values is not advisable, instead, it is recommended to use one of the relevant convenience functions:

```@docs
GuidedProposals.set_parameters!
GuidedProposals.equalize_obs_params!
GuidedProposals.equalize_law_params!
GuidedProposals.same_entries
GuidedProposals.is_critical_update
```

## [Re-computing the guiding term](@id recompute_guiding_term)
-----------------
Sometimes changing parameter values is all that needs to be done; however, it is not always the case. Often changing parameter values implies that the terms used for computation of the guiding term also need to be updated. This needs to be done whenever any parameter of the auxiliary law or any parameter of an observation has changed.

To actually re-compute the guiding terms and finalize the reparameterization you need to call

```@docs
GuidedProposals.recompute_guiding_term!
```

All changes above are done in-place.
