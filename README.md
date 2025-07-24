# BigRiverMetabolomics.jl

*A Toolbox for Metabolomics Analysis*

`BigRiverMetabolomics.jl` is a package that is designed to be a comprehensive toolbox designed to simplify and accelerate metabolomics data analysis. It contains three different aspects that together form a complete metabolomics analysis pipeline:

- **Preprocessing**: This includes data imputation, normalization, transformation and standardization, all provided with the help of the [`BigRiverJunbi.jl`](https://github.com/senresearch/BigRiverJunbi.jl) package.
- **Statistical Analysis**: This includes functions for performing statistical analyses, using both supervised approaches like matrix methods (provided by [`MatrixLM.jl`](https://github.com/senresearch/MatrixLM.jl)) and unsupervised approaches like PCA or PLSDA.
- **Visualization**: This includes functions for visualizing the results of the statistical analyses, using the [`BigRiverMakie.jl`](https://github.com/senresearch/BigRiverMakie.jl) package.

## Installation

To install BigRiverMetabolomics.jl, you can use Julia's package manager:

```julia
using Pkg
Pkg.add("https://github.com/senresearch/BigRiverMetabolomics.jl")
```

or you can use the built-in package manager mode in the Julia REPL:

```julia
] add https://github.com/senresearch/BigRiverMetabolomics.jl
```

## Roadmap

This is a summary of currently implemented features and features that will likely be implemented in the near future. Please note that not only the progress of the list, but also the structure of the list itself is subject to change. This package is still under development and the API is subject to change.

### Preprocessing
_in `BigRiverJunbi.jl`_

- [x] Imputation
   - [x] Zero
   - [x] Minimum value
   - [x] Half-min
   - [x] Probabilistic minimum
   - [x] Mean
   - [x] Categorical-median
   - [x] KNN
   - [x] Quantile Regression
   - [ ] SVD
- [x] Normalization
   - [x] Total area
   - [x] Probablistic quotient
   - [x] Quantile
   - [x] Huberization
- [x] Transformation
    - [x] log
      - [x] (arbitrary base)
      - [x] constant offset
    - [x] mean centering
- [x] Standardization

### Statistical Analysis

- [ ] PCA
- [ ] PLSDA
- [ ] MatrixLM integration

### Visualization
_in `BigRiverMakie.jl`_

- [ ] Dotplot
- [ ] Heatmap
- [ ] Confidence interval plot
