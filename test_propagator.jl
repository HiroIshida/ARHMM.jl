using Revise
using ARHMM

using StaticArrays
using LinearAlgebra
using Distributions

linprop = LinearPropagator(Diagonal(ones(1)), Diagonal(ones(1)), ones(1))
transition_prob(linprop, [0], [1])
