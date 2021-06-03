"""
This code fits the same gravity model that appears in the Marshall paper.

J. M. Marshall, S. L. Wu, H. M. Sanchez C, S. S. Kiware, M. Ndhlovu, A. L. Ouédraogo, M. B.
Touré, H. J. Sturrock, A. C. Ghani, and N. M. Ferguson, “Mathematical models of human
mobility of relevance to malaria transmission in africa,” Sci. Rep., vol. 8, p. 7713, May 2018.
"""
using Distances
using Distributions
using Random
using Turing
using StatsPlots

r"""
This gravity model is from Eqs. 1 and 2 from the Marshall paper.
"""
gravity_model(α, ρ, τ, Nj, dij) = Nj^τ * (one(dij) + dij / ρ)^(-α)
params = Dict(:alpha => 3.62, :rho => exp(5.90), :tau => 0.86)

K = 100
# K is the number of sites, so K - 1 is the number of destinations.
rng = Random.MersenneTwister(9283742)
locations = rand(rng, Float64, (2, K))
distance_matrix = Distances.pairwise(
    Distances.Euclidean(), locations, locations, dims = 2)
Nj = 1000 .+ 100*randn(rng, K)

# This will fit a gravity model. Just that.
p = zeros(Float64, K - 1)
for gen_idx in 1:K-1
    p[gen_idx] = gravity_model(
        3.62, 0.2, 1, Nj[gen_idx + 1], distance_matrix[gen_idx + 1, 1])
end
p = p ./ sum(p)
trips = rand(rng, Multinomial(10000, p))


@model function gravity_model(x = missing, ::Type{T} = Float64) where {T <: Real}
    if x === missing
        x = TArray(T, K - 1)
    end
    s ~ InverseGamma(2, 3)  # error on observation.
    a ~ Gamma(2, 3)
    r ~ Gamma(2, 0.2)
    k = TArray(T, K - 1)
    for kidx in 1:K - 1
        k[kidx] = Nj[kidx] * (one(T) + distance_matrix[kidx + 1, 1] / r)^(-a)
    end
    ktotal = sum(k)

    for dest_idx in 1:K - 1
        x[dest_idx] ~ Normal(1000 * k[dest_idx] / ktotal, 1000 * sqrt(s))
    end
    return s, a, r
end


chn = sample(gravity_model(trips), HMC(0.1, 5), 1000)

describe(chn)

p = plot(chn)
savefig("destination.png")
