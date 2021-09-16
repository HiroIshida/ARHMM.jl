using Revise
using LinearAlgebra
using ARHMM

using StaticArrays
using Distributions

function data_generation(n, A, phi_list, sigma_list)
    x = SVector{1, Float64}([0.0])
    z = 1
    xs = [x]
    zs = [z]
    for i in 1:n
        cat = Categorical(A[:, z])
        z_next = rand(cat)
        x_next = nothing
        if z_next==3
            x_next = [0.0]
        else
            dist = MvNormal(phi_list[z] * x, sigma_list[z])
            x_next = rand(dist)
        end
        x, z = x_next, z_next
        push!(xs, SVector{1, Float64}(x))
        push!(zs, z)
    end
    return xs, zs
end

phi1 = Diagonal(ones(1))
phi2 = Diagonal(ones(1))
phi3 = Diagonal(zeros(1))
sigma1 = Diagonal(ones(1)*0.1)
sigma2 = Diagonal(ones(1)*0.4)
sigma3 = Diagonal(ones(1)*3)

phi_list = [phi1, phi2, phi3]
sigma_list = [sigma1, sigma2, sigma3]

A = [0.9 0. 1.; 
     0.1 0.9 0.0;
     0.0 0.1 0.0]

xs, zs = data_generation(1000, A, phi_list, sigma_list)

hs = HiddenStates(xs, 3)
mp = ModelParameters(1, A, phi_list, sigma_list)
@time ARHMM.update_hidden_states!(hs, mp, xs)

zs_pred = [Float64(argmax(a)) for a in hs.alpha_cache_vec]
using Plots
plot(zs)
plot!(zs_pred)




#@time ARHMM.update_model_parameters!(hs, mp, xs)
