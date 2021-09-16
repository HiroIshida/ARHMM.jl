using Revise
using LinearAlgebra
using ARHMM

using StaticArrays
using Distributions

function data_generation(n, A, prop_list)
    x = SVector{1, Float64}([0.0])
    z = 1
    xs = [x]
    zs = [z]
    for i in 1:n
        cat = Categorical(A[:, z])
        z_next = rand(cat)
        x_next = prop_list[z_next](x)
        x, z = x_next, z_next
        push!(xs, SVector{1, Float64}(x))
        push!(zs, z)
    end
    return xs, zs
end

prop1 = LinearPropagator(Diagonal([1.0]), Diagonal([0.01]), [0.05])
prop2 = LinearPropagator(Diagonal([1.0]), Diagonal([0.01]), [-0.05])
prop3 = FixedPropagator([0.0])
prop_list = [prop1, prop2, prop3]
A = [0.90 0.00 1.0;
     0.10 0.90 0.0;
     0.00 0.10 0.0]
xs, zs = data_generation(300, A, prop_list)

#=
hs = HiddenStates(xs, 2)
mp = ModelParameters(1, A, phi_list, sigma_list)
@time ARHMM.update_hidden_states!(hs, mp, xs)
=#

#=
zs_pred = [Float64(argmax(a)) for a in hs.alpha_cache_vec]
using Plots
plot(zs)
=#

#@time ARHMM.update_model_parameters!(hs, mp, xs)
