using Revise
using ARHMM

using StaticArrays

sequence = [SVector{3, Float64}(0, 0, 0) for _  in 1:1000]
hs = HiddenStates(sequence, 5)
mp = ModelParameters(3, 5)
@time ARHMM.update_hidden_states!(hs, mp, sequence)
