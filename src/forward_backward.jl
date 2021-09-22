function probs_linear_prop(mp::ModelParameters, x_pre, x)
    gen = (transition_prob(prop, x_pre, x) for prop in mp.prop_list)
end

function update_hidden_states!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, xs::Sequence{N}) where {N, M}
    alpha_forward!(hs, mp, xs)
    beta_backward!(hs, mp, xs)
    for t in 1:xs.n_seq - 1
        hs.z_ests[t] = hs.alphas[t] .* hs.betas[t]
    end

    for t in 1:xs.n_seq - 2
        # i is index for t, j for (t+1)
        for i in 1:M
            for j in 1:M
                x_t, x_tt = xs[t+1], xs[t+2]
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

function alpha_forward!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, xs::Sequence{N}) where {N, M}
    hs.c_seq[1] = 1.0

    x1, x2 = xs[1], xs[2]
    px1 = 1.0 # deterministic x 
    tmp = probs_linear_prop(mp, x1, x2) .* mp.pmf_z1 * px1
    hs.c_seq[2] = sum(tmp)
    hs.alphas[1] = tmp/hs.c_seq[2]

    for t in 2:xs.n_seq - 1
        x_t, x_tp1 = xs[t], xs[t+1]
        for i in 1:M
            integral_term = sum(mp.A[i, j] * hs.alphas[t-1][j] for j in 1:M)
            hs.alphas[t][i] = transition_prob(mp.prop_list[i], x_t, x_tp1) * integral_term
        end
        hs.c_seq[t+1] = sum(hs.alphas[t])
        hs.alphas[t] /= hs.c_seq[t+1]
    end
end

function beta_backward!(hs::HiddenStates{M}, mp::ModelParameters{N, M}, xs::Sequence{N}) where {N, M}
    hs.betas[xs.n_seq - 1] = ones(M)
    for t in xs.n_seq-2:-1:1
        x_tp1 = xs[t+1]
        x_tp2 = xs[t+2]
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
