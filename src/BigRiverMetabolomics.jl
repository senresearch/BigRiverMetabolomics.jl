module BigRiverMetabolomics

using BigRiverJunbi
using LinearAlgebra
using Statistics
using StatsBase

# Include analysis functions
include("pca.jl")

# Export functions
export pca_eigen, pca_svd, PCAResult, transform, inverse_transform

end
