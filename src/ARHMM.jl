module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export LinearPropagator, FixedPropagator, transition_prob
include("propagator.jl")

export HiddenStates, ModelParameters, update_model_parameters!, Sequence

struct Sequence{N}
    data::Matrix{Float64}
    n_seq::Int
end
function Sequence(data::Vector)
    n_seq = length(data)
    n_dim = length(data[1])
    mat = zeros(n_dim, n_seq)
    for i in 1:n_seq
        mat[:, i] = data[i]
    end
    Sequence(mat)
end
function Sequence(data::Matrix) 
    n_dim, n_seq = size(data)
    Sequence{n_dim}(data, n_seq)
end
Base.getindex(seq::Sequence, index::Int) = seq.data[:, index]

mutable struct ModelParameters{N, M}
    A::MMatrix{M, M, Float64}
    prop_list::Vector{Propagator{N}}
    pmf_z1::SVector{M, Float64} #TODO should be Dilechlet
end
function ModelParameters(n_dim, A, prop_list)
    n_phase = size(A)[1]
    pmf_z1 = zeros(n_phase); pmf_z1[1] = 1.0 # because initial phase must be phase 1
    ModelParameters{n_dim, n_phase}(A, prop_list, pmf_z1)
end

mutable struct HiddenStates{M}
    n_seq::Int
    z_ests::Vector{MVector{M, Float64}}
    zz_ests::Vector{MMatrix{M, M, Float64}}
    alphas::Vector{MVector{M, Float64}}
    betas::Vector{MVector{M, Float64}}
    c_seq::Vector{Float64}
end
function HiddenStates(n_seq::Int, n_phase::Int)
    M = n_phase
    z_ests = [zeros(M) for _ in 1:n_seq-1]
    zz_ests = [zeros(M, M) for _ in 1:n_seq-2]
    alphas = [MVector{M, Float64}(zeros(M)) for _ in 1:n_seq-1]
    betas = [MVector{M, Float64}(zeros(M)) for _ in 1:n_seq-1]
    c_seq = zeros(n_seq)
    HiddenStates{M}(n_seq, z_ests, zz_ests, alphas, betas, c_seq)
end

function update_pmf_z1!(hs_list::Vector{HiddenStates{M}}, mp::ModelParameters{N, M}, xs_list) where {N, M}
    mp.pmf_z1 = sum(hs.z_ests[1] for hs in hs_list) / length(hs_list)
end

function update_pmf_z1!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, x_seq) where {N, M}
    mp.pmf_z1 = hs.z_ests[1]
end

function update_A!(hs_list::Vector{HiddenStates{M}}, mp::ModelParameters{N, M}, xs_list) where {N, M}
    A_new = zeros(M, M)
    for hs in hs_list
        n_seq = length(hs.z_ests) + 1
        for t in 1:n_seq - 2
            for i in 1:M
                for j in 1:M
                    # Note that our stochastic matrix is different (transposed) from
                    # the one in PRML
                    A_new[j, i] += hs.zz_ests[t][i, j]
                end
            end
        end
    end
    for j in 1:M # normalize
        A_new[:, j] /= sum(A_new[:, j])
    end
    mp.A = A_new
end

function update_A!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, x_seq) where {N, M}
    n_seq = length(hs.z_ests) + 1
    A_new = zeros(M, M)
    for t in 1:n_seq - 2
        for i in 1:M
            for j in 1:M
                # Note that our stochastic matrix is different (transposed) from
                # the one in PRML
                A_new[j, i] += hs.zz_ests[t][i, j]
            end
        end
    end
    for j in 1:M # normalize
        A_new[:, j] /= sum(A_new[:, j])
    end
    mp.A = A_new
end

function update_prop_list!(hs_list::Vector{HiddenStates{M}}, mp::ModelParameters{N, M}, xs_list) where {N, M}
    for i in 1:M
        ws_list = [[z_est[i] for z_est in hs.z_ests] for hs in hs_list]
        fit!(mp.prop_list[i], xs_list, ws_list)
    end
end

function update_model_parameters!(hs_list::Vector{HiddenStates{M}}, mp::ModelParameters{N, M}, xs_list) where {N, M}
    update_pmf_z1!(hs_list, mp, xs_list)
    update_A!(hs_list, mp, xs_list)
    update_prop_list!(hs_list, mp, xs_list)
end

function update_model_parameters!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, xs) where {N, M}
    hs_list = Vector{HiddenStates{M}}([hs])
    xs_list = Vector{Sequence{N}}([xs])
    update_model_parameters!(hs_list, mp, xs_list)
end

export update_hidden_states!
include("forward_backward.jl")

export create_dataset
include("sample_dataset.jl")

end # module
