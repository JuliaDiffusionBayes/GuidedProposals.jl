#==============================================================================#
#
#                 some general & simple utility functions
#
#==============================================================================#

"""
    outer(x)

Compute an outer product
"""
outer(x) = x*x'

Base.lowercase(s::Symbol) = Symbol(lowercase(string(s)))
