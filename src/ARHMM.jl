module ARHMM

using LinearAlgebra
using StaticArrays
using Distributions

export create_dataset
include("sample_dataset.jl")

export LinearPropagator, FixedPropagator, transition_prob
include("propagator.jl")

export HiddenStates, ModelParameters, update_hidden_states!, update_model_parameters!

const Sequence{N} = Vector{SVector{N, Float64}}

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

function probs_linear_prop(mp::ModelParameters, x_pre, x)
    gen = (transition_prob(prop, x_pre, x) for prop in mp.prop_list)
end

function update_hidden_states!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    n_seq = length(seq)
    alpha_forward!(hs, mp, seq)
    beta_backward!(hs, mp, seq)
    for t in 1:n_seq - 1
        hs.z_ests[t] = hs.alphas[t] .* hs.betas[t]
    end
    n_seq = length(seq)

    for t in 1:n_seq - 2
        # i is index for t, j for (t+1)
        for i in 1:M
            for j in 1:M
                x_t, x_tt = seq[t+1:t+2]
                trans_prob = transition_prob(mp.prop_list[j], x_t, x_tt)
                tmp = hs.alphas[t][i] * mp.A[j, i] * trans_prob * hs.betas[t+1][j]
                hs.zz_ests[t][i, j] = tmp/hs.c_seq[t+2]
            end
        end
    end

    # compute log_likelihood
    log_likeli = sum(log(c) for c in hs.c_seq)
    return log_likeli
end

function alpha_forward!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    n_seq = length(seq)
    hs.c_seq[1] = 1.0

    x1, x2 = seq[1:2]
    px1 = 1.0 # deterministic x 
    tmp = probs_linear_prop(mp, x1, x2) .* mp.pmf_z1 * px1
    hs.c_seq[2] = sum(tmp)
    hs.alphas[1] = tmp/hs.c_seq[2]

    for t in 2:n_seq - 1
        x_t, x_tp1 = seq[t:t+1]
        for i in 1:M
            integral_term = sum(mp.A[i, j] * hs.alphas[t-1][j] for j in 1:M)
            hs.alphas[t][i] = transition_prob(mp.prop_list[i], x_t, x_tp1) * integral_term
        end
        hs.c_seq[t+1] = sum(hs.alphas[t])
        hs.alphas[t] /= hs.c_seq[t+1]
    end
end

function beta_backward!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, seq::Sequence{N}) where {N, M}
    n_seq = length(seq)
    hs.betas[n_seq - 1] = ones(M)
    for t in length(seq)-2:-1:1
        x_tp1 = seq[t+1]
        x_tp2 = seq[t+2]
        for j in 1:M # phase at t
            sum = 0.0
            for i in 1:M # phase at t+1
                sum +=mp.A[i, j] * transition_prob(mp.prop_list[i], x_tp1, x_tp2) * hs.betas[t+1][i]
            end
            hs.betas[t][j] = sum
            hs.betas[t][j] /= hs.c_seq[t+2]
        end
   end
end

end # module
