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
using RCall
using Pkg

# Installation of RCall so that it uses the existing installation.
# ENV["R_HOME"]="/usr/lib/R"
# Pkg.build("RCall")

r"""
This gravity model is from Eqs. 1 and 2 from the Marshall paper.
"""
gravity_model(Nj, dij, θ) = Nj^θ[:τ] * (one(dij) + dij / θ[:ρ])^(-θ[:α])
tourist_model(dest_kernel, Nj, dij, Xj, θ) = dest_kernel(Nj, dij, θ) * exp(θ[:β] * Xj)

# params = (β = 0.3, α = 3.62, ρ = exp(5.90), τ = 0.86)
params = (β = 0.3, α = 3.62, ρ = 0.1, τ = 1.0)

xp = 0:0.01:1
yp = map(x -> gravity_model(1, x, params), xp)
# plot(xp, yp)
# plot(xp, pdf.(Distributions.Beta(2, 2), xp))

rng = Random.MersenneTwister(9283742)

r"""
Makes points in a [0, 1] x [0, 1] square according to a process.
If you ask for K points, you will usually get fewer points back.
This returns a (2, N) array of floats.
It matters if our locations are sometimes too close to each other.
It may matter if they are correlated. This is how we use the excellent
functions in R's spatstat to generate those spatial point processes.
"""
function hard_sphere_process(K, dcore = 0.02)
    @rput K
    @rput dcore
    rloc = R"""
        library(spatstat)
        rHardcore(K, dcore)
        """
    x = rcopy(rloc[:x])
    y = rcopy(rloc[:y])
    transpose(hcat(x, y))
end

K_requested = 100
# K is the number of sites, so K - 1 is the number of destinations.
# locations = rand(rng, Float64, (2, K))
locations = hard_sphere_process(K_requested, 0.02)
K = size(locations, 2)
distance_matrix = Distances.pairwise(
    Distances.Euclidean(), locations, locations, dims = 2)
Nj = 1000 .+ 100*randn(rng, K)
Xj = rand(rng, Uniform(-1, 1), K)

# This will fit a gravity model. Just that.
p = zeros(Float64, K - 1)
for gen_idx in 1:K-1
    p[gen_idx] = tourist_model(
        gravity_model, Nj[gen_idx + 1], distance_matrix[gen_idx + 1, 1], Xj[gen_idx + 1], params
        )
end
p .= p ./ sum(p)
# Take a look at the largest probabilities
hcat(p, distance_matrix[2:end, 1])[sortperm(p, rev = true), :]
draws = 10000
trips = rand(rng, Multinomial(draws, p))


@model function inference_model(x = missing, ::Type{T} = Float64) where {T <: Real}
    if x === missing
        x = TArray(T, K - 1)
    end
    # u14 = Uniform(1, 4)
    # to_u14 = inv(bijector(u14))
    # a ~ transformed(u14)
    # α = to_u14(a)
    α ~ Turing.Uniform(1, 4)
    # ur = Uniform(0.05, 0.3)
    # to_ur = inv(bijector(ur))
    # r ~ transformed(ur)
    # ρ = to_ur(r)
    ρ ~ Turing.Uniform(0.05, 3)
    β ~ Turing.Uniform(0.1, 0.5)
    k = Turing.TArray(T, K - 1)
    for kidx in 1:K - 1
        # k[kidx] = Nj[kidx + 1] * (one(T) + distance_matrix[kidx + 1, 1] / ρ)^(-α)
        k[kidx] = tourist_model(gravity_model,
            Nj[kidx + 1], distance_matrix[kidx + 1, 1], Xj[kidx + 1], (β = β, α = α, ρ = ρ, τ = 1.0))
    end
    ktotal = sum(k)

    for dest_idx in 1:K - 1
        x[dest_idx] ~ Turing.Poisson(draws * k[dest_idx] / ktotal)
    end
    return α, ρ, β
end

sample_cnt = 1000
# prior_chain = sample(inference_model, Prior(), sample_cnt)
# posterior_chain = sample(inference_model(missing), HMC(0.1, 5), sample_cnt)
# chn = sample(inference_model(trips), HMC(0.1, 5), sample_cnt)
chn = sample(inference_model(trips), NUTS(0.65), sample_cnt)

describe(chn)
0.9216 * 3 + 1
0.05 + 0.3 * 0.2

p = plot(chn)
savefig("destination.png")
