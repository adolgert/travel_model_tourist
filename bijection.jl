"""
I'm trying to figure out the Bijectors.jl package so that I can use
HMC for sampling in Turing.jl.
"""
using Bijectors
using StatsPlots

dist = Beta(2, 2)
xp = 0:0.01:1.0
plot(xp, pdf.(dist, xp))
# The link function is from distribution's support -> value in R.
y = link(dist, 0.000001)
y = link(dist, 0.999999)

invlink(dist, 13.8)
invlink(dist, -13.8)

b = bijector(dist)
b(0.000001)
b(0.99999)

td = transformed(dist)
rand(td, 20)
itd = inv(b)
itd.(rand(td, 20))


ud = Uniform(0.05, 0.4)
tud = transformed(ud)
rand(tud, 20)
itud = inv(bijector(ud))
itud(-1.25)
itud(-100)
itud(100)