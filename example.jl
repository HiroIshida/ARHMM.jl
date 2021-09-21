using Revise
using Random
using LinearAlgebra
using ARHMM

using StaticArrays
using Distributions

Random.seed!(0)
states_list, phases_list = create_dataset(80)
index = 22

dim = 4
seq = states_list[index]
if dim==2
    xs = [SVector{dim, Float64}(seq[t].x) for t in 1:length(seq)]
elseif dim==4
    xs = [SVector{dim, Float64}(vcat(seq[t].x, seq[t+1].x)) for t in 1:length(seq)-1]
else
    xs = [SVector{dim, Float64}(vcat(seq[t].x, seq[t+1].x, seq[t+2].x)) for t in 1:length(seq)-2]
end
zs = phases_list[index]

prop1 = LinearPropagator(Diagonal(ones(dim)), Diagonal(ones(dim) * 1.0), zeros(dim))
prop2 = LinearPropagator(Diagonal(ones(dim)), Diagonal(ones(dim) * 1.0), zeros(dim))
prop_list = [prop1, prop2]
A = [0.99 0.0;
     0.01 1.0]
mp = ModelParameters(dim, A, prop_list)

z_ests = nothing
log_likelis = []
for _ in 1:200 # if more than 2, usually likelihood will be static
    global z_ests
    z_ests, zz_ests, log_likeli = compute_hidden_states(mp, xs)
    update_model_parameters!(mp, z_ests, zz_ests, xs)
    push!(log_likelis, log_likeli)
end
zs_est = [z[1] for z in z_ests]
println("finish")
using Plots
plot(zs_est)
plot!(zs)
