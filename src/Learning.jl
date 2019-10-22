"""
    All learning algorithms assume a single value function!!!
"""

using Flux
import LinearAlgebra: dot
import Statistics: mean


_square(x::AbstractArray) = x.*x
_prod(x::Array{T, 1}, y::Array{T, 1}) where {T<:Number} = x.*y
_prod(x::Array{T, 1}, y::AbstractArray) where {T<:Number} = x.*y

tderror(v_t::Array{<:AbstractFloat, 1}, c::Array{<:AbstractFloat, 1}, γ_tp1::Array{<:AbstractFloat, 1}, ṽ_tp1::Array{<:AbstractFloat, 1}) where {A<:AbstractArray} = v_t .- (c .+ γ_tp1.*ṽ_tp1)
tderror(v_t, c, γ_tp1, ṽ_tp1) = v_t - (c + γ_tp1*ṽ_tp1)

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

update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0) =
    update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal)


abstract type AbstractTDUpdate <: LearningUpdate end

mutable struct BatchTD <: AbstractTDUpdate end

function update!(model, opt, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0f0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    loss = offpolicy_tdloss(ρ.*corr_term, v_t, r, γ, Flux.data.(v_tp1))
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
    model.W .+= -Flux.Optimise.apply!(opt, model.W, corr_term*Δ)
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

function update!(model::SparseLayer, opt::RMSProp, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)

    dvdt = [deriv(model, s) for s in s_t]
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    fill!(model.ϕ, 0.0f0)
    Δ = δ.*dvdt.*1//length(ρ)
    for i in 1:length(ρ)
        model.ϕ[s_t[i]] .+= corr_term*Δ[i]
    end
    feats = unique(collect(Iterators.flatten(s_t)))

    acc = get!(opt.acc, model.W, zero(model.W))::typeof(model.W)
    acc .*= opt.rho
    acc[feats] .+= (1-opt.rho) .* model.ϕ[feats].^2
    model.W[feats] .-= model.ϕ[feats].*(opt.eta./sqrt.(acc[feats] .+ 1e-8))

end

function update!(model::TabularLayer, opt::Descent, lu::BatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    # println(model, s_t)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = corr_term.*δ.*(1.0/length(v_t))
    for (s_idx, s) in enumerate(s_t)
        model.W[s...] -= opt.eta*Δ[s_idx]
    end
end

struct IncTD <: AbstractTDUpdate end

function update!(model::SparseLayer,
                 opt::Descent,
                 lu::IncTD,
                 ρ, s_t, s_tp1,
                 r, γ, terminal;
                 corr_term=1.0)

    for i in 1:length(ρ)
        v_t = model(s_t[i])
        v_tp1 = model(s_tp1[i])
        dvdt = deriv(model, s_t[i])
        δ = ρ[i]*tderror(v_t, r[i], γ[i], v_tp1)
        model.W[s_t[i]] .-= opt.eta*corr_term*δ*dvdt
    end

end


mutable struct WISBatchTD{T<:AbstractTDUpdate} <: LearningUpdate
    batch_td::T
end

function update!(model, opt, lu::WISBatchTD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    wis_sum = sum(ρ) + 1e-8
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=(1.0/wis_sum)*corr_term*length(ρ))
end

function update!(model, opt, lu::WISBatchTD, ρ::Array{T,1}, s_t, s_tp1, r, γ, terminal; corr_term=1.0) where {T<:AbstractArray}
    wis_sum = (sum(ρ) .+ 1e-8)
    # println(typeof(wis_sum))
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=T(length(ρ)*corr_term./wis_sum))
end

mutable struct WISBatchTD_Rupam <: LearningUpdate
    z::IdDict
    u::IdDict
    d::IdDict
    v::IdDict
    prev_model::IdDict
    u_0::Float64
    normalize_eta::Bool
    WISBatchTD_Rupam(u_0, normalize_eta=false) = new(IdDict(), IdDict(), IdDict(), IdDict(), IdDict(), u_0, normalize_eta)
end

_many_hot(type::Type, size, x::Array{Int, 1}) =
    begin; t = zeros(type, size); t[x] .= one(type); return t; end;

_many_hot(size, x) = _many_hot(Float64, size, x)

function update!(model::SparseLayer, opt::Descent, lu::WISBatchTD_Rupam, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)

    # println(s_t)
    # z_vec = get!(lu.z, model, zeros(length(model.W)))
    # u_vec = get!(lu.u, model, zeros(length(model.W)))
    d_vec = get!(lu.d, model, lu.u_0.*length(s_t[1]).*ones(length(model.W)))::Array{Float64, 1}
    # v_vec = get!(lu.v, model, zeros(length(model.W)))
    
    # println("Step Update")
    for i in 1:length(ρ)
        ϕ = _many_hot(length(model.W), s_t[i])
        # @show ϕ
        ϕϕ = ϕ
        ϕnext = _many_hot(length(model.W), s_tp1[i])
        R = r[i]
        g = 0
        λ = 0
        λnext = 0
        gnext = γ[i]
        rho = ρ[i]
        η = opt.eta
        if lu.normalize_eta
            η = opt.eta/(lu.u_0*length(s_t[1]))
        end

        d_vec .= d_vec - η.*(ϕϕ).*d_vec + rho.*ϕϕ
        dtemp = copy(d_vec)
        dtemp[dtemp.==0.0] .= 1
        alpha = 1 ./ dtemp
        αϕ = alpha[s_t[i]].*ϕ[s_t[i]]
        prednext = model(s_tp1[i])

        v_old = 0.0
        if model ∈ keys(lu.prev_model)
            v_old = lu.prev_model[model](s_t[i])
        end

        model.W[1,s_t[i]] .= model.W[1,s_t[i]] .+ rho.*((R + gnext*prednext - v_old) + (v_old - model(s_t[i]))).*αϕ
        
        if model ∈ keys(lu.prev_model)
            lu.prev_model[model].W .= model.W
        else
            lu.prev_model[model] = deepcopy(model)
        end
    end

    # @show sum(model.W)
    
end

function update!(model::TabularLayer, opt::Descent, lu::WISBatchTD_Rupam, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)

    # println(s_t)
    z_vec = get!(lu.z, model, zeros(length(model.W)))
    u_vec = get!(lu.u, model, zeros(length(model.W)))
    d_vec = get!(lu.d, model, lu.u_0.*ones(length(model.W)))
    v_vec = get!(lu.v, model, zeros(length(model.W)))
    
    # println("Step Update")
    for i in 1:length(ρ)
        ϕ = _many_hot(length(model.W), [s_t[i]])
        # @show ϕ
        ϕϕ = ϕ.*ϕ
        ϕnext = _many_hot(length(model.W), [s_tp1[i]])
        R = r[i]
        g = 0
        λ = 0
        λnext = 0
        gnext = γ[i]
        rho = ρ[i]
        η = opt.eta

        d_vec .= d_vec - η.*(ϕϕ).*d_vec + rho*ϕϕ
        dtemp = copy(d_vec)
        dtemp[dtemp.==0.0] .= 1
        alpha = 1 ./ dtemp
        αϕ = alpha.*ϕ
        # println(minimum(αϕ), " ", maximum(αϕ))
        v_vec .= rho.*ϕϕ
        z_vec .= rho.*αϕ
        prednext = model(s_tp1[i])

        v_old = 0.0
        if model ∈ keys(lu.prev_model)
            v_old = lu.prev_model[model](s_t[i])
        end
        # @show (rho, gnext, R, η)
        # @show maximum(αϕ)
        # @show maximum((R + gnext*prednext - v_old)*z_vec + rho*(v_old - model(s_t[i]))*αϕ)
        
        # @show (R + gnext*prednext - v_old)
        # @show maximum(rho*(v_old - model(s_t[i]))*αϕ)
        # @show size(model.W)
        res_W = reshape(model.W, length(model.W))


        model.W .= model.W + ((R + gnext*prednext - v_old)*z_vec + rho*(v_old - model(s_t[i]))*αϕ)
        lu.prev_model[model] = deepcopy(model)
    end

    # @show sum(model.W)
    
end

# function update!(model::SparseLayer, opt::Descent, lu::WISBatchTD_Rupam, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
#     v_t = model.(s_t)
#     v_tp1 = model.(s_tp1)
#     dvdt = [deriv(model, s) for s in s_t]
#     # δ = ρ.*tderror(v_t, r, γ, v_tp1)
#     # Δ = δ.*dvdt.*1//length(ρ)



#     u_1 = zeros(length(lu.u_vec))
#     u_2 = zeros(length(lu.u_vec))
#     for i in 1:length(ρ)
#         fill!(u_1, 0.0)
#         fill!(u_2, 0.0)

#         v_old = zero(v_t)
#         if model ∈ keys(lu.prev_model)
#             v_old = lu.prev_model[model].(s_t)
#         end
        
#         δ = ρ.*(tderror(v_old, r, γ, v_tp1) + v_old - v_t)
#         Δ = -δ
#         # u_1[s_t[i]] .+= 1 - opt.eta
#         u_1[s_t[i]] .+= opt.eta
#         u_2[s_t[i]] .+= ρ[i]

#         lu.u_vec .= (1 .- u_1).*lu.u_vec .+ u_2
#         # lu.u_vec .= (1 .- u_1.*(1//length(ρ))).*lu.u_vec .+ u_2.*(1//length(ρ))

#         # println(minimum(lu.u_vec[s_t[i]]), " ", maximum(lu.u_vec[s_t[i]]))
#         # println(minimum(1 ./ (lu.u_vec[s_t[i]])), " ", maximum(1 ./ (lu.u_vec[s_t[i]])))

#         # @show v_t
#         # @show r
#         # for i in 1:length(ρ)
#             # println(length(corr_term*Δ[i] ./ lu.u_vec[s_t[i]]))
#         model.W[s_t[i]] .+= corr_term*Δ[i] ./ (lu.u_vec[s_t[i]] .+ 1e-8)
#         # println(Δ[i])
#         # println(minimum(Δ[i] ./ lu.u_vec[s_t[i]]), " ", maximum(Δ[i] ./ lu.u_vec[s_t[i]]))
#         # end
#         lu.prev_model[model] = deepcopy(model)
#     end
#     # lu.u_vec .= (u_1.*(1//length(ρ))).*lu.u_vec .+ u_2.*(1//length(ρ))

#     println(minimum(model.W), " ", maximum(model.W))

# end


mutable struct VTrace{T<:AbstractTDUpdate} <: LearningUpdate
    ρ_bar::Float64
    batch_td::T
end

# VTrace(ρ_bar, tdupdate) = new(ρ_bar, tdupdate)

function update!(model, opt, lu::VTrace, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    update!(model, opt, lu.batch_td, clamp.(ρ, 0.0, lu.ρ_bar), s_t, s_tp1, r, γ, terminal; corr_term=corr_term)
end

function update!(model, opt, lu::VTrace, ρ::Array{Array{T, 1}, 1}, s_t, s_tp1, r, γ, terminal; corr_term=1.0)  where {T<:AbstractFloat}
    clamp_ρ = [T.(clamp.(_ρ, T(0.0f0), T(lu.ρ_bar))) for _ρ in ρ]
    update!(model, opt, lu.batch_td, clamp_ρ, s_t, s_tp1, r, γ, terminal; corr_term=T(corr_term))
end

mutable struct IncNormIS{T<:AbstractTDUpdate} <: LearningUpdate
    max_is::IdDict
    batch_td::T
end

IncNormIS(tdupdate) = IncNormIS(IdDict(), tdupdate)

function update!(model, opt, lu::IncNormIS, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    if !(model ∈ keys(lu.max_is))
        lu.max_is[model] = 1.0f0
    end
    max_is = max(lu.max_is[model], maximum(ρ))
    lu.max_is[model] = max_is
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=corr_term/max_is)
end

function update!(model, opt, lu::IncNormIS, ρ::Array{Array{T, 1}, 1}, s_t, s_tp1, r, γ, terminal; corr_term=1.0f0)  where {T<:AbstractFloat}
    if !(model ∈ keys(lu.max_is))
        lu.max_is[model] = ones(T, length(ρ[1]))
    end
    max_is = lu.max_is[model]
    for idx in 1:length(max_is)
        max_is[idx] = max(max_is[idx], maximum(getindex.(ρ, idx)))
    end
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=corr_term./max_is)
end

mutable struct WSNormIS{T<:AbstractTDUpdate} <: LearningUpdate
    beta::Float32
    weighted_sum_is::IdDict
    batch_td::T
end

WSNormIS(beta::Float32, tdupdate) = WSNormIS(beta, IdDict(), tdupdate)

function update!(model, opt, lu::WSNormIS, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    get!(lu.weighted_sum_is, model, 0.0f0)::Float32
    lu.weighted_sum_is[model] = Float32(lu.beta*maximum(ρ)^2 + (1.0f0-lu.beta)*lu.weighted_sum_is[model])
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=corr_term/sqrt(lu.weighted_sum_is[model]))
end

function update!(model, opt, lu::WSNormIS, ρ::Array{Array{T, 1}, 1}, s_t, s_tp1, r, γ, terminal; corr_term=1.0f0)  where {T<:AbstractFloat}
    weighted_sum_is = get!(lu.weighted_sum_is, model, zeros(T, length(ρ[1])))
    for idx in 1:length(weighted_sum_is)
        weighted_sum_is[idx] = lu.beta*maximum(getindex.(ρ, idx))^2 + (1.0f0-lu.beta)*weighted_sum_is[idx]
    end
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=corr_term./sqrt.(weighted_sum_is))
end

mutable struct WSAvgNormIS{TD<:AbstractTDUpdate} <: LearningUpdate
    beta::Float32
    weighted_sum_is::IdDict
    batch_td::TD
end

WSAvgNormIS(beta::Float32, tdupdate) = WSAvgNormIS(beta, IdDict(), BatchTD())

function update!(model, opt, lu::WSAvgNormIS, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    get!(lu.weighted_sum_is, model, 0.0f0)::Float32
    lu.weighted_sum_is[model] = Float32(lu.beta*mean(ρ)^2 + (1.0f0-lu.beta)*lu.weighted_sum_is[model])
    update!(model, opt, lu.batch_td, ρ, s_t, s_tp1, r, γ, terminal; corr_term=corr_term/sqrt(lu.weighted_sum_is[model]))
end

function update!(model, opt, lu::WSAvgNormIS, ρ::Array{Array{T, 1}, 1}, s_t, s_tp1, r, γ, terminal; corr_term=1.0f0)  where {T<:AbstractFloat}
    weighted_sum_is = get!(lu.weighted_sum_is, model, zeros(T, length(ρ[1])))::type(Array{T, 1})
    for idx in 1:length(max_is)
        weighted_sum_is[idx] = lu.beta*mean(getindex.(ρ, idx))^2 + (1.0f0-lu.beta)*weighted_sum_is[idx]
    end
    update!(model, opt, lu.batch_td, clamp_ρ, s_t, s_tp1, r, γ, terminal; corr_term=corr_term./sqrt.(weighted_sum_is))
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

function update!(model::TabularLayer, opt::Descent, lu::BatchSarsa, ρ, s_t::Array{Int64, 1}, s_tp1::Array{Int64, 1}, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0)

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

function update!(model::TabularLayer, opt::Descent, lu::BatchExpectedSarsa, ρ, s_t::Array{Int64, 1}, s_tp1::Array{Int64, 1}, r, γ, terminal, a_t, a_tp1, target_policy::Array{Float64, 1}; corr_term=1.0)
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




