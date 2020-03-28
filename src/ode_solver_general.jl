#==============================================================================#
#
#   General definitions shared by all ODE system solvers (H,F,c system, M,L,μ
#   system and P,ν system)
#
#==============================================================================#

"""
    AbstractGuidingTermSolver{Tmode}

Supertype for ODE solvers (solving H,F,c system or M,L,μ system or P,ν system).
`Tmode` is a flag for whether computations are done in-place (with states
represented by vectors), out-of-place (with state represented by StaticArrays),
or on GPUs (with states represented by cuArrays).
"""
abstract type AbstractGuidingTermSolver{Tmode} end

mode(::AbstractGuidingTermSolver{Tmode}) where Tmode = Tmode
