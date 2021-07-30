using Turing
using StatsPlots


# The input data is a set of trips and lengths of trips for a
# number of people at each home place. It's also the distance
# among home places.
# We can modify it to be a number of trips to each place

@model function destination_model(x)
end


chn = sample(destination_model(a), HMC(0.1, 5), 1000)

describe(chn)

p = plot(chn)
savefig("destination.png")
