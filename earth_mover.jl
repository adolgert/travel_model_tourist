using Random
"""
Earth-mover distance for sites on a graph.

`a` is the current distribution and `b` is the one we move to.
Both are vectors of Reals. `r` is the distance of each site.
"""
function earth_mover_discrete(a::Vector{Float64}, b::Vector{Float64}, r::Vector{Float64})
    T = Float64
    total = zero(T)
    i = length(a)
    j = length(b)  # Everything after j is correct.
    while j > 0
        c = zero(T)  # Represents amount of a at jth site.
        while i > 0 && a[i] < b[j] - c
            c += a[i]
            total += a[i] * abs(r[j] - r[i])
            a[i] = zero(T)
            i -= 1
        end
        if i > 0
            Δ = b[j] - c
            a[i] -= Δ
            c += Δ
            total += Δ * abs(r[j] - r[i])
        end
        j -= 1
    end
    return total
end


function test_emd_smoke()
    rng = Random.MersenneTwister(923847234)
    for trial in 1:100
        N = 20
        distances = rand(rng, N)
        sort!(distances)
        travels = 327.4 * rand(rng, N)
        desired = shuffle(rng, travels)
        total = earth_mover_discrete(travels, desired, distances)
        @assert(total > 0)
    end
end

function test_emd_none()
    rng = Random.MersenneTwister(62937498)
    for trial in 1:100
        N = 20
        distances = rand(rng, N)
        sort!(distances)
        travels = 327.4 * rand(rng, N)
        desired = copy(travels)
        total = earth_mover_discrete(travels, desired, distances)
        @assert(total < 1e-9)
    end
end


function test_emd_single_shift_forward()
    N = 20
    distances = zeros(Float64, N)
    distances .= 1:N
    sort!(distances)
    travels = ones(Float64, N)
    desired = copy(travels)
    mid = N>>1
    desired[mid + 3] += desired[mid]
    desired[mid] = 0
    total = earth_mover_discrete(travels, desired, distances)
    @assert(abs(total - 3.0) < 1e-9)
end

function test_emd_single_shift_backward()
    N = 20
    distances = zeros(Float64, N)
    distances .= 1:N
    sort!(distances)
    travels = ones(Float64, N)
    desired = copy(travels)
    mid = N>>1
    desired[mid - 4] += desired[mid]
    desired[mid] = 0
    total = earth_mover_discrete(travels, desired, distances)
    @assert(abs(total - 4.0) < 1e-9)
end
