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
        x_next = prop_list[z_next](x)
        x, z = x_next, z_next
        push!(xs, SVector{1, Float64}(x))
        push!(zs, z)
    end
    return xs, zs
end

#Random.seed!(2)
prop1 = LinearPropagator(Diagonal([1.0]), Diagonal([0.01]), [0.1])
prop2 = LinearPropagator(Diagonal([1.0]), Diagonal([0.01]), [-0.1])
prop_list = [prop1, prop2]
A = [0.95 0.05;
     0.05 0.95]

xs, zs = data_generation(30, A, prop_list)
hs = HiddenStates(xs, 2)
mp = ModelParameters(1, A, prop_list)
ARHMM.alpha_forward!(hs, mp, xs)
ARHMM.beta_backward!(hs, mp, xs)

cat_pred = [(a .* b)/sum(a .* b) for (a, b) in zip(hs.beta_seq, hs.alpha_seq )]
zs_pred = [argmax(z) for z in cat_pred]

using Plots
plot(zs)
plot!(zs_pred)
