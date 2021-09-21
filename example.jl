using Revise
using Random
using LinearAlgebra
using ARHMM

using StaticArrays
using Distributions

Random.seed!(0)
states_list, phases_list = create_dataset(80)
index = 8

seq = states_list[index]
xs = [SVector{4, Float64}(vcat(seq[t].x, seq[t+1].x)) for t in 1:length(seq)-1]
zs = phases_list[index]

prop1 = LinearPropagator(Diagonal(ones(4)), Diagonal(ones(4) * 1.0), zeros(4))
prop2 = LinearPropagator(Diagonal(ones(4)), Diagonal(ones(4) * 1.0), zeros(4))
prop_list = [prop1, prop2]
A = [0.99 0.0;
     0.01 1.0]
mp = ModelParameters(4, A, prop_list)

z_ests = nothing
log_likelis = []
for _ in 1:200 # if more than 2, usually likelihood will be static
    global z_ests
    z_ests, zz_ests, log_likeli = compute_hidden_states(mp, xs)
    update_model_parameters!(mp, z_ests, zz_ests, xs)
    push!(log_likelis, log_likeli)
end
zs_est = [z[1] for z in z_ests]
using Plots
plot(zs_est)
plot!(zs)
