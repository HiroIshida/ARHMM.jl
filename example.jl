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

Random.seed!(2)
prop1 = LinearPropagator(Diagonal([1.0]), Diagonal([0.1^2]), [0.1])
prop2 = LinearPropagator(Diagonal([1.0]), Diagonal([0.1^2]), [-0.1])
prop_list = [prop1, prop2]
A = [0.95 0.05;
     0.05 0.95]

xs, zs = data_generation(50, A, prop_list)
mp = ModelParameters(1, A, prop_list)
alpha_seq = ARHMM.alpha_forward(mp, xs)
beta_seq = ARHMM.beta_backward(mp, xs)

cat_pred = [(a .* b)/sum(a .* b) for (a, b) in zip(alpha_seq, beta_seq)]
zs_pred = [argmax(z) for z in cat_pred]

using Plots
plot(zs)
plot!(zs_pred)
