using LinearAlgebra
using Distributions
using StaticArrays

abstract type Propagator{N} end

struct LinearPropagator{N} <: Propagator{N}
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

struct FixedPropagator{N} <: Propagator{N}
    fixed_point::SVector{N, Float64}
end
FixedPropagator(fixed_point::AbstractVector) = FixedPropagator{length(fixed_point)}(fixed_point)
transition_prob(prob::FixedPropagator, x_before, x_after, eps=1e-6) = 1.0 * (norm(prob.fixed_point - x_after) < eps)
(prop::FixedPropagator)(x) = prop.fixed_point
