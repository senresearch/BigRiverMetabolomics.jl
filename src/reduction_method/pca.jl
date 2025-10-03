"""
Holds a fitted PCA model.
- `components`  (p×k): loadings (principal axes)
- `scores`      (n×k): PC scores
- `eigenvalues` (k)  : variances along PCs
- `explained_variance_ratio` (k)
- `mean` (p)
- `std` (p)
- `weights` (`AbstractWeights` used)
- `neff` : effective sample size ≈ (Σw)^2 / Σ(w^2)
"""
struct PCAResult{T}
    components::Matrix{T}
    scores::Matrix{T}
    eigenvalues::Vector{T}
    explained_variance_ratio::Vector{T}
    mean::Vector{T}
    std::Vector{T}
    weights::AbstractWeights
    neff::Float64
end

# Effective sample size for any AbstractWeights
function _neff(w::AbstractWeights)
    v = StatsBase.weights(w)
    return (sum(v)^2) / sum(abs2, v)
end

# columnwise weighted mean / std (unbiased std)
function _wcolmean(X::AbstractMatrix, w::AbstractWeights)
    n, p = size(X)
    μ = similar(float.(X[1, :]), p)
    @inbounds for j in 1:p
        μ[j] = mean(view(X, :, j), w)
    end
    return μ
end

function _wcolstd(Xc::AbstractMatrix, w::AbstractWeights)
    n, p = size(Xc)
    σ = similar(float.(Xc[1, :]), p)
    @inbounds for j in 1:p
        s = std(view(Xc, :, j), w; corrected = true)
        σ[j] = (s == 0 || !isfinite(s)) ? one(eltype(s)) : s
    end
    return σ
end

# Shared preprocessing: returns (Xs, μ, σ, w, neff)
function _prep(
        X::AbstractMatrix;
        center::Bool, scale::Bool,
        weights::Union{Nothing, AbstractWeights}
    )
    Xf = float.(X)
    n, p = size(Xf)
    @assert n > 1 "PCA requires at least 2 rows."

    w = weights === nothing ? FrequencyWeights(fill(one(eltype(Xf)), n)) : weights
    neff = _neff(w)
    @assert neff > 1.0 "Effective sample size must exceed 1."

    μ = center ? _wcolmean(Xf, w) : zeros(eltype(Xf), p)
    Xc = center ? Xf .- reshape(μ, 1, :) : Xf
    σ = scale ? _wcolstd(Xc, w) : ones(eltype(Xf), p)
    Xs = Xc ./ reshape(σ, 1, :)

    return Xs, μ, σ, w, neff
end

"""
    pca_eigen(X; k=nothing, center=true, scale=false, whiten=false, weights=nothing)

PCA via eigen-decomposition of the (weighted) covariance/correlation matrix.
"""
function pca_eigen(
        X::AbstractMatrix; k::Union{Int, Nothing} = nothing,
        center::Bool = true, scale::Bool = false, whiten::Bool = false,
        weights::Union{Nothing, AbstractWeights} = nothing
    )

    Xs, μ, σ, w, neff = _prep(X; center, scale, weights)

    # Weighted covariance (unbiased per StatsBase semantics)
    S = Symmetric(cov(Xs, w; corrected = true))  # denom depends on weight type

    F = eigen(S)
    ord = sortperm(F.values; rev = true)
    vals, vecs = F.values[ord], F.vectors[:, ord]

    p = size(Xs, 2)
    k === nothing && (k = p)
    k = min(k, p)

    components = vecs[:, 1:k]
    scores = Xs * components
    if whiten
        scores = scores * Diagonal(1 ./ sqrt.(max.(vals[1:k], eps(eltype(vals)))))
    end

    ev = vals[1:k]
    evratio = ev ./ sum(vals)

    return PCAResult{eltype(Xs)}(components, scores, ev, evratio, μ, σ, w, neff)
end

"""
    pca_svd(X; k=nothing, center=true, scale=false, whiten=false, weights=nothing)

PCA via SVD of the √weight-scaled, centered(/scaled) data. This matches the
variance scaling of `pca_eigen` (unbiased `cov(…, w; corrected=true)`).

Implementation notes:
- Form `Z = diagm(√w) * Xs` (done via row scaling with `sqrt.(weights(w))`).
- If `s` are singular values of `Z`, the *biased* weighted covariance eigenvalues
  would be `s.^2 / sum(w)`. We rescale to *unbiased* using the ratio
  `trace(cov(Xs, w; corrected=true)) / trace(cov(Xs, w; corrected=false))`
  so the result matches `pca_eigen` regardless of weight type.
"""
function pca_svd(
        X::AbstractMatrix; k::Union{Int, Nothing} = nothing,
        center::Bool = true, scale::Bool = false, whiten::Bool = false,
        weights::Union{Nothing, AbstractWeights} = nothing
    )

    Xs, μ, σ, w, neff = _prep(X; center, scale, weights)

    v = StatsBase.weights(w)
    sumw = sum(v)
    Z = Xs .* reshape(sqrt.(v), :, 1)   # row-scale by √weights

    U, s, V = svd(Z; full = false)             # Z ≈ U*Diagonal(s)*V'
    p = size(Xs, 2)
    k === nothing && (k = p)
    k = min(k, p)

    # Convert singular values -> unbiased covariance eigenvalues
    ev_biased = (s .^ 2) ./ sumw
    Sb_trace = tr(cov(Xs, w; corrected = false))
    Su_trace = tr(cov(Xs, w; corrected = true))
    scale_ratio = Sb_trace > 0 ? (Su_trace / Sb_trace) : one(eltype(s))
    vals = ev_biased .* scale_ratio

    # Sort by variance
    ord = sortperm(vals; rev = true)
    components = V[:, ord[1:k]]
    ev = vals[ord[1:k]]

    scores = Xs * components
    if whiten
        scores = scores * Diagonal(1 ./ sqrt.(max.(ev, eps(eltype(ev)))))
    end
    evratio = ev ./ sum(vals)

    return PCAResult{eltype(Xs)}(components, scores, ev, evratio, μ, σ, w, neff)
end

"""
    transform(Xnew::AbstractMatrix, fit::PCAResult)

Project new data using a fitted PCA (same centering/scaling).
"""
function transform(Xnew::AbstractMatrix, fit::PCAResult)
    Xf = float.(Xnew)
    Xs = (Xf .- reshape(fit.mean, 1, :)) ./ reshape(fit.std, 1, :)
    return Xs * fit.components
end

"""
    inverse_transform(Z::AbstractMatrix, fit::PCAResult)

Approximately reconstruct data from scores back to the original feature space.
"""
function inverse_transform(Z::AbstractMatrix, fit::PCAResult)
    Xs_approx = Z * fit.components'
    return Xs_approx .* reshape(fit.std, 1, :) .+ reshape(fit.mean, 1, :)
end
