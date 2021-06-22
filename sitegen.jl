r"""
Generate sites for travel.
"""
using Distributions
using RCall
using Pkg
using Random

r"""
Makes points in a [0, 1] x [0, 1] square according to a process.

K is the number of points. `core_fraction` is the fraction of the total area that should
be used as a hard core. In other words, if there are 100 points in the area, then the core
around those points is 100π r^2, which should be a fraction of the total area. The total
area here is always 1. We parameterize this way so that we don't accidentally ask for lots
more hard core space than we can support.

If you ask for K points, you will usually get fewer points back but can also get too many
points back, because that's how spatstat's sampling works.
This returns a (2, N) array of floats.
It matters if our locations are sometimes too close to each other.
It may matter if they are correlated. This is how we use the excellent
functions in R's spatstat to generate those spatial point processes.
"""
function hard_sphere_process(K, core_fraction = 0.02)
    dcore = sqrt(core_fraction / (K * pi))
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

size(hard_sphere_process(1000, 0.02), 2)


r"""
Generate sites according to Zipf's law.

Zipf's law says that city sizes will follow a Pareto distribution.
A nice article is "Zipf's law for all the natural cities in the United States:
a geospatial perspective," by Bin Jiang and Tau Jia (2010).
The values are α = 1.74, α = 1.91.
"""
function zipf_sites(rng, cnt::Integer, shape::Real, core::Real = 0.02)
    locations = hard_sphere_process(cnt, core)
    K = size(locations, 2)
    
    population_distribution = Pareto(shape)
    pops = rand(rng, population_distribution, K)
    (locations, pops)
end


function test_zipf_site_ratios()
    rng = Random.MersenneTwister(2497234)
    locs, pops = zipf_sites(rng, 100, 1.7)

    shape = 1.74
    scale = 1000000
    cnt = 1000
    ratios = zeros(Float64, cnt)
    for i in 1:cnt
        population_distribution = Pareto(shape, scale)
        pops = rand(rng, population_distribution, cnt)
        big_to_little = maximum(pops) / minimum(pops)
        ratios[i] = big_to_little
    end
    println(median(ratios))
end
