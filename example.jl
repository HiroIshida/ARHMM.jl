using Revise
using Random
using LinearAlgebra
using ARHMM

using StaticArrays
using Distributions

Random.seed!(0)
states_list, phases_list = create_dataset(80)
xs = [SVector{2, Float64}(state.x) for state in states_list[1]]
zs = phases_list[1]

prop1 = LinearPropagator(Diagonal(ones(2)), Diagonal(ones(2) * 100.0), zeros(2))
prop2 = LinearPropagator(Diagonal(ones(2)), Diagonal(ones(2) * 100.0), zeros(2))
prop_list = [prop1, prop2]
A = [0.95 0.05;
     0.05 0.95]
mp = ModelParameters(2, A, prop_list)

z_ests = nothing
for _ in 1:100 # if more than 2, usually likelihood will be static
    global z_ests
    z_ests, zz_ests, log_likeli = compute_hidden_states(mp, xs)
    update_model_parameters!(mp, z_ests, zz_ests, xs)
    #println(mp.A)
    #println(mp.prop_list.A)
end

# TODO it is strange that when A[2, 2] = 0.0, it does not save the relation

