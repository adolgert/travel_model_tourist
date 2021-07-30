"""
This one isn't running! Not done.

Going to take a travel model and take a tourist index,
then distribute people. This file takes a distribution and
relaxes it until it has a minimum earth mover's distance from the
travel model, while still obeying the tourist index.
"""

using Distances
using Random
using Distributions
using VoronoiDelaunay
using Turing
using StatsPlots

include("sitegen.jl")
include("earth_mover.jl")


r"""
This gravity model is from Eqs. 1 and 2 from the Marshall paper.
Get the type from the parameter because that's what carries the implicit differentiation type.
Nj is population at site j. dij is distance i to j. θ are parameters.
"""
gravity_model(Nj, dij, θ) = Nj^θ[:τ] * (one(θ[:ρ]) + dij / θ[:ρ])^(-θ[:α])

# The tourist/gravity models aren't normalized. They are a likelihood to go to a destination.
# This calculates likelihoods across all destinations and then normalizes.
# source is origin site. Nj is population at sites. Xj is tourist covariate at sites.
# distance matrix is from i to j. θ is a dictionary, or named tuple, of parameters.
function kernel_flux!(travel_matrix, Nj, distance_matrix, θ)
    K = length(Nj)
    p = zeros(Float64, K)
    for col in 1:K  # The source is col.
        for gen_idx in 1:K  # The destination is row.
            # No if-then on the source because its distance to itself is infinite.
            p[gen_idx] = gravity_model(
                Nj[gen_idx], distance_matrix[gen_idx, col], θ
                )
        end
        travel_matrix[:, col] = Nj[col] * θ[:b] * p ./ sum(p)
    end
end

function build_distances(locations)
    distance_matrix = Distances.pairwise(
        Distances.Euclidean(), locations, locations, dims = 2)
        # Set self-trips to have an infinite distance because kernels behave well at infinity, not zero.
    for dmidx in 1:size(distance_matrix, 2)
        distance_matrix[dmidx, dmidx] = Inf
    end
    distance_matrix
end

rng = MersenneTwister(294723)
N_desired = 5
total_population = 10_000
# β is tourist rate. α is exponent for gravity. ρ is kernel distance
# for gravity model. τ is exponent on population for gravity.
# zα is shape for distribution of city sizes.
# b is fraction of people who travel during specified time.
params = (β = 0.3, α = 3.62, ρ = 0.1, τ = 1.0, zα = 1.74, b = 0.4)
locations, pops_raw = zipf_sites(rng, N_desired, params[:zα])
N = length(pops_raw)
distance_matrix = build_distances(locations)
Nj = pops_raw .* (total_population ./ sum(pops_raw))
flux_matrix = similar(distance_matrix)
kernel_flux!(flux_matrix, Nj, distance_matrix, params)
total_travelers = sum(flux_matrix)
tourist_tendency = Nj .* exp.(randn(rng, N))
tourist_flux = total_travelers .* tourist_tendency ./ sum(tourist_tendency)
# So now we have a total traveling and a total being visited.

K = N - 1
vpriors = zeros(Float64, (K, N))
dist = Uniform(1, 5)
for init_idx in 1:N
    for xidx in 1:K
        vpriors[xidx, init_idx] = rand(rng, dist)
    end
    vpriors[:, init_idx] = vpriors[:, init_idx] ./ sum(vpriors[:, init_idx])
end

people = [100, 20, 200, 50, 40, 70]
@assert length(people) == N
total_travelers = sum(people)

small = 0.05
# Treat these as probabilities for a _single traveler._ It's a one-electron universe.
@model function persons(x)
    p1 ~ Dirichlet(vpriors[:, 1])
    p2 ~ Dirichlet(vpriors[:, 2])
    p3 ~ Dirichlet(vpriors[:, 3])
    p4 ~ Dirichlet(vpriors[:, 4])
    p5 ~ Dirichlet(vpriors[:, 5])
    p6 ~ Dirichlet(vpriors[:, 6])

    x[1] ~ Normal(p2[1] + p3[1] +p4[1] + p5[1] + p6[1], small)
    x[2] ~ Normal(p1[1] + p3[2] +p4[2] + p5[2] + p6[2], small)
    x[3] ~ Normal(p1[2] + p2[2] +p4[3] + p5[3] + p6[3], small)
    x[4] ~ Normal(p1[3] + p2[3] +p3[3] + p5[4] + p6[4], small)
    x[5] ~ Normal(p1[4] + p2[4] +p3[4] + p4[4] + p6[5], small)
    x[6] ~ Normal(p1[5] + p2[5] +p3[5] + p4[5] + p5[5], small)
end

suggest = people
observed = N .* suggest ./ sum(suggest)
sample_cnt = 1000
chn = sample(persons(observed), NUTS(0.65), sample_cnt)

describe(chn)

p = plot(chn)
