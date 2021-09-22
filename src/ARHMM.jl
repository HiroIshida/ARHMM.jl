module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export LinearPropagator, FixedPropagator, transition_prob
include("propagator.jl")

export HiddenStates, ModelParameters, Sequence
include("types.jl")

export update_model_parameters!
include("maximization.jl")

export update_hidden_states!
include("forward_backward.jl")

export create_dataset
include("sample_dataset.jl")

end # module
