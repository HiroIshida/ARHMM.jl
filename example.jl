using Revise
using Random
using LinearAlgebra
using ARHMM

using StaticArrays
using Distributions

Random.seed!(0)
states_list, phases_list = create_dataset(80)
index = 10
xs = [SVector{2, Float64}(state.x) for state in states_list[index]]
zs = phases_list[index]

prop1 = LinearPropagator(Diagonal(ones(2)), Diagonal(ones(2) * 1.0), zeros(2))
prop2 = LinearPropagator(Diagonal(ones(2)), Diagonal(ones(2) * 1.0), zeros(2))
prop_list = [prop1, prop2]
A = [0.95 0.0;
     0.05 1.0]
mp = ModelParameters(2, A, prop_list)

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
