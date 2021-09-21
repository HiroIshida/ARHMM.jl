using LinearAlgebra
using Distributions
using StaticArrays

abstract type Propagator{N} end

mutable struct LinearPropagator{N} <: Propagator{N}
    phi::SMatrix{N, N, Float64}
    cov::SMatrix{N, N, Float64}
    drift::SVector{N, Float64}
end
function LinearPropagator(phi, cov, drift)
    N = size(phi)[1]
    phi_ = SMatrix{N, N, Float64}(phi)
    cov_ = SMatrix{N, N, Float64}(cov)
    drift_ = SVector{N, Float64}(drift)
    LinearPropagator{N}(phi_, cov_, drift_)
end

function fit!(prop::LinearPropagator, x_seq, w_seq)
    n = length(w_seq)
    x_sum = sum(x_seq[t] * w_seq[t] for t in 1:n-1)
    y_sum = sum(x_seq[t+1] * w_seq[t] for t in 1:n-1)
    xx_sum = sum(x_seq[t] * x_seq[t]' * w_seq[t] for t in 1:n-1)
    xy_sum = sum(x_seq[t] * x_seq[t+1]' * w_seq[t] for t in 1:n-1)
    w_sum = sum(w_seq[1:n-1])

    # Thanks to Gauss-markov theorem, we can separate fitting processes into
    # first, non probabilistic term  
    phi_est = inv(w_sum * xx_sum - x_sum * x_sum') * (w_sum * xy_sum - x_sum * y_sum')
    b_est = (y_sum - phi_est' * x_sum) * (1.0/w_sum)

    # and covariance part
    self_cross(vec) = vec * vec'
    cov_est = sum(w_seq[t] * self_cross(x_seq[t+1] - (phi_est * x_seq[t] + b_est)) for t in 1:n-1)/w_sum

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
