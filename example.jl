using Revise
using ARHMM

using StaticArrays

sequence = [SVector{3, Float64}(0, 0, 0) for _  in 1:100]
hs = HiddenStates(sequence, 5)
mp = ModelParameters(3, 5)
ARHMM.alpha_forward!(hs, mp, sequence)

