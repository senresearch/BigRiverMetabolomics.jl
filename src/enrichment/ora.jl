"""
    ora_results(predictor_pval, myZgroups, mysubsets, dfMetabolo, dfAnnotation; thresh = 0.05)

Perform overrepresentation analysis (ORA) using the hypergeometric distribution.

# Arguments
- `predictor_pval`: Vector of p-values for each metabolite.
- `myZgroups`: Vector associating each metabolite with a specific subset or group.
- `mysubsets`: Vector of unique identifiers corresponding to the subsets of interest.
- `dfMetabolo`: DataFrame where each column represents a metabolite.
- `dfAnnotation`: DataFrame containing annotations; must include a column `lipID` corresponding to metabolite identifiers.
- `thresh`: Significance threshold (default = 0.05). Metabolites with p-values below this threshold are considered significant.

# Returns
A DataFrame with two columns:
- `Subset`: The name of the subset.
- `Pval`: The hypergeometric p-value indicating the significance of the overrepresentation of significant metabolites in that subset.
"""
function ora_results(predictor_pval, myZgroups, mysubsets, dfMetabolo, dfAnnotation; thresh = 0.05)

    # Initialize the results DataFrame with 'Subset' names and placeholder for p-values.
    dfORA = DataFrame(
        Subset = mysubsets, 
        Pval = Vector{Float64}(undef, length(mysubsets))
    );

    # Identify indices of metabolites with p-values below the threshold (i.e., significant metabolites).
    idx_sgnf_tg = findall(predictor_pval .< thresh)
    
    # Retrieve the names of significant metabolites from the DataFrame's column names.
    set_sgnf_tg = names(dfMetabolo)[idx_sgnf_tg]

    # Calculate the total number of metabolites (population size) based on the columns of dfMetabolo.
    size_metabolites = size(dfMetabolo, 2)
    
    # Determine the total number of significant metabolites in the entire population.
    size_significant = length(idx_sgnf_tg)

    # Loop through each subset specified in mysubsets.
    for i in 1:length(mysubsets)
        # Identify indices of metabolites that belong to the current subset (using group assignments).
        idx_subset = findall(myZgroups .== mysubsets[i])
        
        # Retrieve the corresponding metabolite identifiers from the annotation DataFrame.
        subset_tg = dfAnnotation.lipID[idx_subset]

        # Calculate the size of the current subset (i.e., the number of metabolites in the subset).
        size_subset = length(idx_subset)
        
        # Count the number of significant metabolites in the current subset by computing the intersection.
        size_significant_subset = length(intersect(set_sgnf_tg, subset_tg))

        # Define the hypergeometric distribution parameters:
        # - Number of successes in the population: size_significant
        # - Number of failures in the population: size_metabolites - size_significant
        # - Sample size: size_subset
        mydist = Hypergeometric(
            size_significant,        
            size_metabolites - size_significant,
            size_subset
        )
        
        # Compute the p-value using the complementary cumulative distribution function (ccdf)
        # which estimates the probability of observing 'size_significant_subset' or more successes.
        dfORA.Pval[i] = ccdf(mydist, size_significant_subset)
    end

    # Return the DataFrame with overrepresentation results (subset names and corresponding p-values).
    return dfORA
end