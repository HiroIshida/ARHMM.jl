using LinearAlgebra
using Distributions
using StaticArrays

abstract type Propagator{N} end

mutable struct LinearPropagator{N} <: Propagator{N}
    phi::Matrix{Float64}
    cov::Matrix{Float64}
    drift::Vector{Float64}
end
function LinearPropagator(phi, cov, drift)
    N = size(phi)[1]
    @assert size(cov) == (N, N)
    @assert size(phi) == (N, N)
    @assert size(drift) == (N, )
    LinearPropagator{N}(phi, cov, drift)
end

function fit!(prop::LinearPropagator{N}, xs_list, ws_list) where N
    x_sum = zeros(N)
    y_sum = zeros(N)
    xx_sum = zeros(N, N)
    xy_sum = zeros(N, N)
    w_sum = 0.0

    for (xs, ws) in zip(xs_list, ws_list)
        x_sum += sum(xs[t] * ws[t] for t in 1:length(xs)-1)
        y_sum += sum(xs[t+1] * ws[t] for t in 1:length(xs)-1)
        xx_sum += sum(xs[t] * xs[t]' * ws[t] for t in 1:length(xs)-1)
        xy_sum += sum(xs[t] * xs[t+1]' * ws[t] for t in 1:length(xs)-1)
        w_sum += sum(ws[1:end-1])
    end

    # Thanks to Gauss-markov theorem, we can separate fitting processes into
    # first, non probabilistic term  
    phi_est = inv(w_sum * xx_sum - x_sum * x_sum') * (w_sum * xy_sum - x_sum * y_sum')
    b_est = (y_sum - phi_est' * x_sum) * (1.0/w_sum)

    # and covariance part
    self_cross(vec) = vec * vec'
    cov_est = zeros(N, N)
    cov_est = sum(
                sum(ws[t] * self_cross(xs[t+1] - (phi_est * xs[t] + b_est)) for t in 1:length(xs)-1)/w_sum 
                for (xs, ws) in zip(xs_list, ws_list))

    prop.phi = phi_est
    prop.drift = b_est
    prop.cov = cov_est
end

function transition_prob(prop::LinearPropagator, x_before, x_after)
    mean = prop.phi * x_before + prop.drift
    dist = MvNormal(mean, Matrix(prop.cov))
    pdf(dist, x_after)
end

function (prop::LinearPropagator)(x)
    mean = prop.phi * x + prop.drift
    dist = MvNormal(mean, Matrix(prop.cov))
    rand(dist)
end

mutable struct FixedPropagator{N} <: Propagator{N}
    fixed_point::SVector{N, Float64}
end
FixedPropagator(fixed_point::AbstractVector) = FixedPropagator{length(fixed_point)}(fixed_point)
transition_prob(prob::FixedPropagator, x_before, x_after, eps=1e-6) = 1.0 * (norm(prob.fixed_point - x_after) < eps)
(prop::FixedPropagator)(x) = prop.fixed_point
