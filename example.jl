using Revise
using Random
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
        x_next = prop_list[z](x)
        x, z = x_next, z_next
        push!(xs, SVector{1, Float64}(x))
        push!(zs, z)
    end
    return xs, zs
end

Random.seed!(0)
prop1 = LinearPropagator(Diagonal([1.0]), Diagonal([0.1^2]), [0.4])
prop2 = LinearPropagator(Diagonal([1.0]), Diagonal([0.1^2]), [-0.4])
prop_list = [prop1, prop2]
A = [0.85 0.15;
     0.15 0.85]

xs, zs = data_generation(3000, A, prop_list)
A_pred_init = [0.5 0.5;
               0.5 0.5]
mp = ModelParameters(1, A_pred_init, prop_list)
z_ests = emrun!(mp, xs, 20)
z_preds = [argmax(z) for z in z_ests]

println("done")
using Plots
T = 500
plot(zs[1:T])
plot!(z_preds[1:T])
