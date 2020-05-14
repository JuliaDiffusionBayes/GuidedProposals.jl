# [Computations of log-likelihoods](@id log_likelihood_computations)
*********************************
The full likelihood function for a sampled path $X$ given by:
```math
\frac{
    \color{forestgreen}{
        \tilde{\rho}(0,X_0)
    }
}{
    \color{maroon}{
        \rho(0,X_0)
    }
}\exp\left\{
    \color{royalblue}{
        \int_0^T G(t, X_t) \dd t
    }
\right\},
```
where
```math
G(t,x):=\left[
    (b-\tilde{b})^T\tilde{r}
    + 0.5 tr\left\{
        (a-\tilde{a})(\tilde{r}\tilde{r}^T-H)
    \right\}
\right](t,x),
```
and $\tilde{r}(t,x):=\nabla\log\tilde{\rho}(t,x)$. The terms $\color{royalblue}{\int_0^T G(t, X_t) \dd t}$, and $\log\left(\color{forestgreen}{\tilde{\rho}(0,X_0)}\right)$ may be computed with functions:
```@docs
GuidedProposals.loglikhd
```
and
```@docs
GuidedProposals.loglikhd_obs
```
respectively. In general, deriving the term $\color{maroon}{\rho(0,X_0)}$ explicitly is impossible. Thankfully though, in an MCMC or an importance sampling setting this term always cancels out and so never needs to be computed.

## Log-likelihood computation whilst sampling
--------------------
Function `rand!`—when called with a parameter `Val(:ll)`—computes the "log-likelihood" at the time of sampling. Internally the following function is called after the Wiener process is sampled.
```@docs
GuidedProposals.solve_and_ll!
```
`solve_and_ll!` computes only $\color{royalblue}{\int_0^T G(t, X_t) \dd t}$.
- When `rand!` is called on a **single** `GuidProp` (i.e. a single interval) then only this path contribution is returned.
- However, if `rand!` is called on **a list of** `GuidProp`, then apart from summing over the results from `solve_and_ll!` an additional end-point contribution is added, i.e.
```math
\color{royalblue}{\int_0^T G(t, X_t) \dd t}+\log\left(\color{forestgreen}{\tilde{\rho}(0,X_0)}\right)
```
is returned.
