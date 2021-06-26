"""
This version fits the following data:

1. Location of every site.
2. Population of every site.
3. Observed number of trips from one site to all others.
4. Total number of trips, from all other sites, to each site.

That last part is the tourist portion and makes this a quadratic problem.
The gravity model is from the Marshall paper.

J. M. Marshall, S. L. Wu, H. M. Sanchez C, S. S. Kiware, M. Ndhlovu, A. L. Ouédraogo, M. B.
Touré, H. J. Sturrock, A. C. Ghani, and N. M. Ferguson, “Mathematical models of human
mobility of relevance to malaria transmission in africa,” Sci. Rep., vol. 8, p. 7713, May 2018.
"""

using Distances
using Distributions
using Random
using Turing
using StatsPlots
using RCall
using Pkg

include("sitegen.jl")

# Installation of RCall so that it uses the existing installation.
# ENV["R_HOME"]="/usr/lib/R"
# Pkg.build("RCall")

r"""
This gravity model is from Eqs. 1 and 2 from the Marshall paper.
"""
# Get the type from the parameter because that's what carries the implicit differentiation type.
gravity_model(Nj, dij, θ) = Nj^θ[:τ] * (one(θ[:ρ]) + dij / θ[:ρ])^(-θ[:α])
tourist_model(dest_kernel, Nj, dij, Xj, θ) = dest_kernel(Nj, dij, θ) * exp(θ[:β] * Xj)


function show_gravity_cutoff_smaller_than_region(θ)
    xp = 0:0.01:1
    yp = map(x -> gravity_model(1, x, θ), xp)
    plot(xp, yp)
end


# This will fit a gravity model. Just that.
function probability_leave_site(source, Nj, Xj, distance_matrix, θ)
    p = zeros(Float64, K)
    for gen_idx in 1:K
        # No if-then on the source because its distance to itself is infinite.
        p[gen_idx] = tourist_model(
            gravity_model, Nj[gen_idx], distance_matrix[gen_idx, source], Xj[gen_idx], θ
            )
    end
    p .= p ./ sum(p)
end


# This returns an aggregate number of inbound trips to each site from all other sites.
# It is a rate per year, per person. You can think of it as the total inbound trips if each
# person at each site took one trip per year. Multiply by a rate of trips per year, as desired.
function tourist_trips!(inbound_trips, rate, Nj, Xj, distance_matrix, θ)
    T = typeof(θ[:α])
    K = length(inbound_trips)
    p = zeros(T, K)
    for source in 1:K
        p .= zero(T)
        for gen_idx in 1:K
            p[gen_idx] = tourist_model(
                gravity_model, Nj[gen_idx], distance_matrix[gen_idx, source], Xj[gen_idx], θ
                )
        end
        # This normalization means that trips in one direction are different from trips in other.
        inbound_trips .+= rate * Nj[source] .* p ./ sum(p)
    end
end

# plot(xp, pdf.(Distributions.Beta(2, 2), xp))

rng = Random.MersenneTwister(9283742)
# K is the number of sites, so K - 1 is the number of destinations.
# params = (β = 0.3, α = 3.62, ρ = exp(5.90), τ = 0.86)

params = (β = 0.3, α = 3.62, ρ = 0.1, τ = 1.0, zα = 1.74)

K_requested = 30
total_population = 10000
# Locations are in [0, 1] x [0, 1].
# Smallest pops are greater than 1. Largest pops are near 20 or 200.
locations, pops_raw = zipf_sites(rng, K_requested, params[:zα])
K = size(locations, 2)
Nj = pops_raw .* (total_population ./ sum(pops_raw))
# Tourism covariate.
Xj = rand(rng, Distributions.Normal(0, 0.5), K)

distance_matrix = Distances.pairwise(
    Distances.Euclidean(), locations, locations, dims = 2)
# Set self-trips to have an infinite distance because kernels behave well at infinity, not zero.
for dmidx in 1:K
    distance_matrix[dmidx, dmidx] = Inf
end

# Take a look at the largest probabilities
# hcat(p, distance_matrix[2:end, 1])[sortperm(p, rev = true), :]
nth_largest_site = 3
data_site = (1:K)[sortperm(vec(Nj), rev=true)][nth_largest_site]

p = probability_leave_site(data_site, Nj, Xj, distance_matrix, params)
draws = 10000
rate_multiplier = draws / Nj[data_site]

outbound_trips = convert(Vector{Float64}, rand(rng, Multinomial(draws, p)))
inbound_trips = zeros(Float64, K)
tourist_trips!(inbound_trips, rate_multiplier, Nj, Xj, distance_matrix, params)


"""
x is the number of trips from the data site to other sites.
r is the number of visiting trips to _all sites_ over that time period.
"""
@model all_sites_model(x = missing, r = missing, ::Type{T} = Float64) where {T} = begin
    α ~ Turing.Uniform(1, 4)
    ρ ~ Turing.Uniform(0.05, 3)
    β ~ Turing.Uniform(0.1, 0.5)

    if x === missing
        x = TArray(T, K)
        println("not x missing")
    end
    if r === missing
        r = TArray(T, K)
        println("not r missing")
    end
    println("x $(typeof(x)) r $(typeof(r)) α $(typeof(α)) T $(T)")

    inr = zeros(typeof(α), K)
    tourist_trips!(
        inr, rate_multiplier, Nj, Xj, distance_matrix,
        (β = β, α = α, ρ = ρ, τ = params[:τ])
    )
    for inbound_idx in 1:K
        r[inbound_idx] ~ Normal(inr[inbound_idx], 10)
    end

    k = Turing.TArray(T, K)
    for kidx in 1:K
        k[kidx] = tourist_model(
            gravity_model, Nj[kidx], distance_matrix[kidx, data_site], Xj[kidx],
            (β = β, α = α, ρ = ρ, τ = params[:τ])
            )
    end
    ktotal = sum(k)

    for dest_idx in 1:K
        x[dest_idx] ~ Turing.Poisson(draws * k[dest_idx] / ktotal)
    end
    return α, ρ, β
end

sample_cnt = 1000
# prior_chain = sample(all_sites_model, Prior(), sample_cnt)
# posterior_chain = sample(all_sites_model(missing), HMC(0.1, 5), sample_cnt)
# chn = sample(all_sites_model(trips), HMC(0.1, 5), sample_cnt)
chn = sample(all_sites_model(outbound_trips, inbound_trips), NUTS(0.65), sample_cnt)

describe(chn)

p = plot(chn)
savefig("destination.png")
