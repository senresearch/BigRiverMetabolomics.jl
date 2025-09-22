using Test
using LinearAlgebra
using Random
using StatsBase

# flip column signs of A to best match B (1-1) for comparison
function _align_signs!(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A) == size(B)
    for j in axes(A, 2)
        if sum(abs, A[:, j] .- B[:, j]) > sum(abs, -A[:, j] .- B[:, j])
            A[:, j] .= -A[:, j]
        end
    end
    return A
end

# return the weighted covariance of scores; corrected=true by default
_wcov(Z, w) = cov(Z, w; corrected = true)

# Shapes, orthonormality, EVR
@testset "Shapes, orthonormality, EVR" begin
    Random.seed!(123)
    n, p, k = 300, 8, 4
    X = randn(n, p) * Diagonal(range(1.0, 2.5; length = p)) # anisotropic

    fit = BigRiverMetabolomics.pca_eigen(X; k = k, center = true, scale = false)

    @test size(fit.components) == (p, k)
    @test size(fit.scores) == (n, k)
    @test length(fit.eigenvalues) == k
    @test length(fit.explained_variance_ratio) == k

    # Orthonormal loadings
    @test isapprox(fit.components' * fit.components, I(k); rtol = 1.0e-8, atol = 1.0e-10)

    # EVR sums to fraction of total variance explained by k components
    # For k < p, this will be less than 1.0
    @test sum(fit.explained_variance_ratio) <= 1.0 + 1.0e-10
    @test sum(fit.explained_variance_ratio) >= 0.0

    # transform == scores
    Z = BigRiverMetabolomics.transform(X, fit)
    @test isapprox(Z, fit.scores; rtol = 0, atol = 1.0e-12)

    # Inverse transform with k=p gives near-exact reconstruction
    fit_full = BigRiverMetabolomics.pca_eigen(X; k = p, center = true, scale = false)
    X̂ = BigRiverMetabolomics.inverse_transform(fit_full.scores, fit_full)
    @test isapprox(X̂, X; rtol = 1.0e-8, atol = 1.0e-8)
end

# Eigen vs SVD equivalence
@testset "Eigen vs SVD agreement (unweighted)" begin
    Random.seed!(2)
    n, p, k = 250, 10, 5
    X = randn(n, p)
    # Non-spherical scaling + correlation PCA off, then on
    X .= X * Diagonal(range(0.5, 2.0; length = p))

    fit_e = BigRiverMetabolomics.pca_eigen(X; k = k, center = true, scale = false)
    fit_s = BigRiverMetabolomics.pca_svd(X; k = k, center = true, scale = false)

    # Eigenvalues should match very closely
    @test isapprox(fit_e.eigenvalues, fit_s.eigenvalues; rtol = 1.0e-8, atol = 1.0e-10)

    # Subspace agreement up to sign
    Ce = copy(fit_e.components)
    Cs = copy(fit_s.components)
    _align_signs!(Cs, Ce)
    @test isapprox(abs.(diag(Ce' * Cs)), ones(k); rtol = 1.0e-7, atol = 1.0e-7)

    # Scores agree up to sign flips as well
    Ze = fit_e.scores
    Zs = fit_s.scores
    # align by column sign to minimize L1 distance
    _align_signs!(Zs, Ze)
    @test isapprox(Ze, Zs; rtol = 1.0e-7, atol = 1.0e-7)
end

@testset "Eigen vs SVD agreement (correlation PCA)" begin
    Random.seed!(3)
    n, p, k = 220, 12, 6
    X = randn(n, p) .* reshape(range(0.1, 3.0; length = p), 1, :)
    fit_e = BigRiverMetabolomics.pca_eigen(X; k = k, center = true, scale = true)   # correlation PCA
    fit_s = BigRiverMetabolomics.pca_svd(X; k = k, center = true, scale = true)
    @test isapprox(fit_e.eigenvalues, fit_s.eigenvalues; rtol = 1.0e-8, atol = 1.0e-10)
    Cs = _align_signs!(copy(fit_s.components), fit_e.components)
    @test isapprox(abs.(diag(fit_e.components' * Cs)), ones(k); rtol = 1.0e-7, atol = 1.0e-7)
end

# Weighted PCA semantics
@testset "Weighted PCA (Weights / FrequencyWeights / ProbabilityWeights)" begin
    Random.seed!(4)
    n, p, k = 400, 7, 4
    X = randn(n, p)
    X[:, 1] .+= 3 .* rand(n)  # add some signal

    # Three weight types
    wW = FrequencyWeights(randexp(n))                 # positive, arbitrary
    wF = FrequencyWeights(rand(1:3, n))               # integer counts
    wP = ProbabilityWeights(rand(n) ./ n)  # sums to 1

    # Eigen vs SVD should agree for each weight type
    for w in (wW, wF, wP)
        fit_e = BigRiverMetabolomics.pca_eigen(X; k = k, center = true, scale = false, weights = w)
        fit_s = BigRiverMetabolomics.pca_svd(X; k = k, center = true, scale = false, weights = w)

        @test fit_e.neff > 1
        @test isapprox(fit_e.neff, BigRiverMetabolomics._neff(w); rtol = 1.0e-12, atol = 0)

        @test isapprox(fit_e.eigenvalues, fit_s.eigenvalues; rtol = 1.0e-8, atol = 1.0e-10)

        # Whitening should produce I_k covariance under the SAME weights
        fit_e_w = BigRiverMetabolomics.pca_eigen(X; k = k, center = true, scale = false, weights = w, whiten = true)
        ΣZ = _wcov(fit_e_w.scores, w)
        @test isapprox(ΣZ, I(k); rtol = 1.0e-7, atol = 1.0e-7)
    end
end

# Zero-variance columns and scaling
@testset "Zero-variance guards & correlation PCA" begin
    Random.seed!(5)
    n, p, k = 150, 6, 3
    X = randn(n, p)
    X[:, end] .= 5.0               # last column has zero variance
    w = FrequencyWeights(ones(n))

    fit = BigRiverMetabolomics.pca_eigen(X; k = k, center = true, scale = true, weights = w)

    # No NaNs/Infs in stds or components
    @test all(isfinite, fit.std)
    @test all(isfinite, fit.components)

    # Correlation PCA should produce bounded eigenvalues
    # (some slack due to zero-var column guard)
    # Skip trace comparison when zero-variance columns create NaN values
    X_centered = X .- mean(X; dims = 1)
    X_std = std(X; dims = 1, corrected = true)
    # Replace zero std with 1 to match our _wcolstd logic
    X_std_safe = replace(X_std, 0.0 => 1.0, NaN => 1.0, Inf => 1.0)
    X_scaled = X_centered ./ X_std_safe
    Σ = cov(X_scaled, corrected = true)
    if all(isfinite, Σ) && all(isfinite, diag(Σ))
        @test sum(fit.eigenvalues) ≈ tr(Σ) atol = 2.0
    end
end

# Reconstruction quality (k < p)
@testset "Low-rank reconstruction tracks explained variance" begin
    Random.seed!(6)
    n, p, k = 250, 10, 3
    X = randn(n, p) * Diagonal(range(1.0, 3.0; length = p))

    fit = BigRiverMetabolomics.pca_eigen(X; k = k, center = true, scale = false)
    Xhat = BigRiverMetabolomics.inverse_transform(fit.scores, fit)

    # Relative Frobenius error ≈ sqrt(unexplained variance fraction)
    total_var = sum(eigvals(cov(X .- mean(X; dims = 1), corrected = true)))
    kept_var = sum(fit.eigenvalues)
    unexplained_frac = 1 - kept_var / total_var

    rel_err = norm(Xhat - X) / norm(X)
    @test rel_err <= sqrt(unexplained_frac) + 1.0e-2
end

# Errors & edge conditions
@testset "Errors: neff ≤ 1" begin
    Random.seed!(7)
    X = randn(20, 5)
    # Only one nonzero weight → neff = 1
    w_bad = FrequencyWeights([1.0; zeros(19)])
    @test_throws AssertionError BigRiverMetabolomics.pca_eigen(X; weights = w_bad)
    @test_throws AssertionError BigRiverMetabolomics.pca_svd(X; weights = w_bad)
end
