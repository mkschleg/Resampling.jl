"""
    All learning algorithms assume a single value function!!!
"""

using Flux
import LinearAlgebra: dot
import Statistics: mean


_square(x::AbstractArray) = x.*x
_prod(x::Array{T, 1}, y::Array{T, 1}) where {T<:Number} = x.*y
_prod(x::Array{T, 1}, y::AbstractArray) where {T<:Number} = x.*y

tderror(v_t, c, γ_tp1, ṽ_tp1) = v_t .- (c .+ γ_tp1.*ṽ_tp1)

function offpolicy_tdloss(ρ_t::Array{Array{T, 1}, 1},
                          v_t::AbstractArray,
                          c::Array{Array{T, 1}, 1},
                          γ_tp1::Array{Array{T,1}, 1},
                          ṽ_tp1::Array{Array{T,1}, 1}) where {T<:AbstractFloat}
    target = c .+ _prod.(γ_tp1, ṽ_tp1)
    error = _square.(v_t .- target)
    return (T(0.5))*sum(mean.(_prod.(ρ_t, error))) * (1//length(ρ_t))
end

tdloss(v_t, c, γ_tp1, ṽ_tp1) =
    0.5*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))


"""
    LearningUpdate
WIP - Currently LearningUpdate and Optimizer are haphazardly similar....
"""
abstract type LearningUpdate end


"""
    train!(value, opt::Optimizer, lu::LearningUpdate, ρ, s_t, s_tp1, reward, γ, terminal)
# Arguments:
`value::ValueFunction`:
`opt::Optimizer`:
`ρ`: Importance sampling ratios (Array of Floats)
`s_t`: States at time t
`s_tp1`: States at time t + 1
`reward`: cumulant or reward for value function
`γ`: discount factor
`terminal`: Determining termination of the episode (if applicable).
"""
# train!(value::AbstractVFunction, lu::LearningUpdate, ϕ_t, ϕ_tp1, reward, γ, ρ, terminal)

"""
    train!(value::ValueFunction, opt::Optimizer, lu::LearningUpdate, ρ, s_t, s_tp1, reward, γ, terminal, a_t, a_tp1, target_policy)
# Arguments:
`value::ValueFunction`:
`opt::Optimizer`:
`ρ`: Importance sampling ratios (Array of Floats)
`s_t`: States at time t
`s_tp1`: States at time t + 1
`reward`: cumulant or reward for value function
`γ`: discount factor
`terminal`: Determining termination of the episode (if applicable).
`a_t`: Action at time t
`a_tp1`: Action at time t + 1
`target_policy`: Action at time t
"""

# train!(value::AbstractQFunction, lu::LearningUpdate, ϕ_t, ϕ_tp1, reward, γ, ρ, terminal, a_t, a_tp1, target_policy)

update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0) =
    update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal)

mutable struct BatchTD <: LearningUpdate end

function update!(model, opt, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    loss = offpolicy_tdloss(ρ, v_t, r, γ, Flux.data.(v_tp1))
    grads = Flux.gradient(()->loss, params(model))
    for weights in params(model)
        Flux.Optimise.update!(opt, weights, grads[weights])
    end
end

function update!(model::SingleLayer, opt, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    dvdt = deriv.(model, s_t)
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = mean(δ.*dvdt.*s_t)
    model.W .+= -apply!(opt, model.W, corr_term*Δ)
end

function update!(model::SparseLayer, opt::Descent, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    dvdt = [deriv(model, s) for s in s_t]
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = δ.*dvdt.*1//length(ρ)

    for i in 1:length(ρ)
        model.W[s_t[i]] .-= opt.eta*corr_term*Δ[i]
    end
end

function update!(model::TabularLayer, opt::Descent, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = corr_term.*δ.*(1.0/length(v_t))
    for (s_idx, s) in enumerate(s_t)
        model.W[s...] -= opt.eta*Δ[s_idx]
    end
end


mutable struct WISBatchTD <: LearningUpdate
    batch_td::BatchTD
    WISBatchTD() = new(BatchTD())
end

function update!(model, opt, lu::WISBatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    wis_avg = mean(ρ)
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=(1.0/wis_avg)*corr_term)
end

function update!(model, opt, lu::WISBatchTD, ρ::Array{T,1}, s_t, s_tp1, r, γ, terminal; corr_term=1.0) where {T<:AbstractArray}
    wis_avg = mean(mean(ρ))
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=(1.0./wis_avg)*corr_term)
end


mutable struct VTrace <: LearningUpdate
    ρ_bar::Float64
    batch_td::BatchTD
    VTrace(ρ_bar) = new(ρ_bar, BatchTD())
end

function update!(model, opt, lu::VTrace, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    update!(model, opt, lu.batch_td, clamp.(ρ, 0.0, lu.ρ_bar), s_t, s_tp1, r, γ, terminal; corr_term=corr_term)
end

function update!(model, opt, lu::VTrace, ρ::Array{Array{T, 1}, 1}, s_t, s_tp1, r, γ, terminal; corr_term=1.0)  where {T<:AbstractFloat}
    clamp_ρ = [clamp.(_ρ, T(0.0), T(lu.ρ_bar)) for _ρ in ρ]
    update!(model, opt, lu.batch_td, clamp_ρ, s_t, s_tp1, r, γ, terminal; corr_term=corr_term)
end

"""
    All Q Learning algorithms are defined with multiple outputs, there is no cont action cases yet.
"""

# Q Learning

TabularQLayer(num_actions, dims...) = TabularLayer(num_actions, dims...)

mutable struct BatchSarsa <: LearningUpdate end

function update!(model, opt, lu::BatchSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    preds_t = [model(s_t[idx])[act] for (idx, act) in enumerate(a_t)]
    preds_tp1 = [model(s_tp1[idx])[act] for (idx, act) in enumerate(a_tp1)]
    grads = Flux.gradient(()->offpolicy_tdloss(preds, r, γ, Flux.data.(preds_tp1)), params(model))
    for weights in params(model)
        update!(opt, weights, -grads[weights])
    end
end

function update!(model::SingleLayer, opt, lu::BatchSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    q_t = [model(s_t[idx])[act] for (idx, act) in enumerate(action_t)]
    q_tp1 = [model(s_tp1[idx])[act] for (idx, act) in enumerate(a_tp1)]
    dvdt = [deriv(model, s_t[idx]) for (idx, act) in enumerate(action_t)]
    δ = tderror(v_t, r, γ, v_tp1)
    Δ = mean(δ.*dvdt.*s_t)
    model.W .+= -apply!(opt, model.W, corr_term*Δ)
end

function update!(model::TabularLayer, opt::Descent, lu::BatchSarsa, ρ, s_t::Array{Array{Int64, 1}, 1}, s_tp1::Array{Array{Int64, 1}, 1}, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)

    cis_t = [CartesianIndex(act, s_t[idx]...) for (idx, act) in enumerate(a_t)]
    q_t = model.(cis_t)
    q_tp1 = [model(CartesianIndex(act, s_tp1[idx]...)) for (idx, act) in enumerate(a_tp1)]
    δ = tderror(q_t, r, γ, q_tp1)
    Δ = corr_term.*δ.*(1.0/length(q_t))
    for (ci_idx, ci) in enumerate(cis_t)
        model[ci] -= opt.eta*Δ[ci_idx]
    end
end

mutable struct BatchExpectedSarsa <: LearningUpdate end

function update!(model, opt, lu::BatchExpectedSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    preds_t = [model(s_t[idx])[act] for (idx, act) in enumerate(a_t)]
    exp_q = [dot(target_policy, model(s_tp1[idx])) for (idx, act) in enumerate(a_tp1)]
    grads = Flux.gradient(()->offpolicy_tdloss(preds, r, γ, Flux.data.(exp_q)), params(model))
    for weights in params(model)
        update!(opt, weights, -grads[weights])
    end
end

function update!(model::SingleLayer, opt, lu::BatchExpectedSarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)
    q_t = [model(s_t[idx])[act] for (idx, act) in enumerate(action_t)]
    exp_q = [dot(target_policy, model(s_tp1[idx])) for (idx, act) in enumerate(a_tp1)]
    dvdt = [deriv(model, s_t[idx]) for (idx, act) in enumerate(action_t)]
    δ = tderror(v_t, r, γ, exp_q)
    Δ = mean(δ.*dvdt.*s_t)
    model.W .+= -apply!(opt, model.W, corr_term*Δ)
end

function update!(model::TabularLayer, opt::Descent, lu::BatchExpectedSarsa, ρ, s_t::Array{Array{Int64, 1}, 1}, s_tp1::Array{Array{Int64, 1}, 1}, r, γ, terminal, a_t, a_tp1, target_policy::Array{Float64, 1}; corr_term=1.0)
    q_t = [model(CartesianIndex(act, s_t[idx]...)) for (idx, act) in enumerate(a_t)]
    exp_q = [dot(target_policy, (model[:, s...])) for s in s_tp1]
    δ = tderror(q_t, r, γ, exp_q)
    Δ = corr_term.*δ.*(1.0/length(q_t))
    for (s_idx, s) in enumerate(s_t)
        model[a_t[s_idx], s...] -= opt.eta*Δ[s_idx]
    end
end

function update!(model::TabularLayer, opt::Descent, lu::BatchExpectedSarsa, ρ, s_t::Array{Array{Int64, 1}, 1}, s_tp1::Array{Array{Int64, 1}, 1}, r, γ, terminal, a_t, a_tp1, target_policy::P; corr_term=1.0) where {P<:AbstractPolicy}
    q_t = [model(CartesianIndex(act, s_t[idx]...)) for (idx, act) in enumerate(a_t)]
    # println([get(target_policy, s, a, nothing, nothing, nothing) for a in 1:size(model.W)[1]])
    exp_q = [dot([get(target_policy, s, a, nothing, nothing, nothing) for a in 1:size(model.W)[1]], (model[:, s...])) for s in s_tp1]
    δ = tderror(q_t, r, γ, exp_q)
    Δ = corr_term.*δ.*(1.0/length(q_t))
    for (s_idx, s) in enumerate(s_t)
        model[a_t[s_idx], s...] -= opt.eta*Δ[s_idx]
    end
end




