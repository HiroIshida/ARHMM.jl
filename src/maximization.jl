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
