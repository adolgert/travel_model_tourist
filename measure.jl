# This takes data from the Marshall et al paper and measures the tourist index of it.
# The data is available as a supplement to the paper.
#
# J. M. Marshall, S. L. Wu, H. M. Sanchez C, S. S. Kiware, M. Ndhlovu, A. L. Ouédraogo, M. B.
# Touré, H. J. Sturrock, A. C. Ghani, and N. M. Ferguson, “Mathematical models of human
# mobility of relevance to malaria transmission in africa,” Sci. Rep., vol. 8, p. 7713, May 2018.

import XLSX
using Geodesy


function find_columns(column_names)
    col_indices = zeros(Int, length(column_names))
    col_idx = 1
    col_name = coordinates[1, col_idx]
    while col_name !== missing
        for (try_idx, try_name) in enumerate(column_names)
            if typeof(try_name) <: Regex
                if occursin(try_name, col_name)
                    col_indices[try_idx] = col_idx
                end
            else
                if try_name == col_name
                    col_indices[try_idx] = col_idx
                end
            end
        end
        col_idx += 1
        col_name = coordinates[1, col_idx]
    end
    col_indices
end


function read_country(xf, country)
    coordinates = xf["$country Coordinates"]
    last_row = XLSX.get_dimension(coordinates).stop.row_number
    llp_cols = find_columns([r"^LON", r"^LAT", r"^POP"])

    location_id = vec(convert(Matrix{Int}, coordinates["A2:A$last_row"]))
    typeassert(location_id[1], Integer)
    lon_lat = zeros(last_row - 1, 3)
    for read_row in 2:last_row
        for read_col in 1:3
            lon_lat[read_row - 1, read_col] = convert(Float64, coordinates[read_row, llp_cols[read_col]])
        end
    end

    trips = xf["$country Trips"]
    last_trip_row = XLSX.get_dimension(trips).stop.row_number
    trip_ij = zeros(Int, last_trip_row - 1, 2)
    for trip_row_idx in 2:last_trip_row
        for trip_col in 1:2
            trip_ij[trip_row_idx - 1, trip_col] = trips[trip_row_idx, trip_col]
        end
    end
    location_id, lon_lat, trip_ij
end

# This file is a paper supplement.
marshall_data = "41598_2018_26023_MOESM2_ESM.xlsx"
xf = XLSX.readxlsx(marshall_data)
XLSX.sheetnames(xf)

country = "Tanzania"
location_id, lon_lat, trip_ij = read_country(xf, "Tanzania")
length(location_id)

function distance_matrix(lon_lat)
    locations = Vector{Geodesy.LLA}(undef, size(lon_lat)[1])
    for parse_idx in 1:length(locations)
        # incoming matrix is longitude-latitude, and we need the opposite, so 2 then 1.
        locations[parse_idx] = Geodesy.LLA(lon_lat[parse_idx, 2], lon_lat[parse_idx, 1], 0)
    end

    M = zeros(Float64, length(locations), length(locations))
    for row in 1:length(locations)
        for col in 1:(row - 1)
            M[row, col] = Geodesy.euclidean_distance(locations[row], locations[col])
            M[col, row] = M[row, col]
        end
    end
    M
end

M = distance_matrix(lon_lat)
N = lon_lat[:, 3]

r"""
This gravity model is from Eqs. 1 and 2 from the Marshall paper.
"""
gravity_model(α, ρ, τ, Nj, dij) = Nj^τ * (one(dij) + dij / ρ)^(-α)

params = Dict(:alpha => 3.62, :rho => exp(5.90), :tau => 0.86)
# Create an un-normalized probability and then normalize it.
Pij = similar(M)
for ui in 1:length(location_id)
    for uj in 1:length(location_id)
        if ui != uj
            Pij[ui, uj] = gravity_model(
                params[:alpha], params[:rho], params[:tau], N[uj], M[ui, uj])
        else
            Pij[ui, uj] = 0
        end
    end
end

Pji = transpose(Pij)
sum_over_destinations = sum(Pji, dims = 2)
for norm_col in 1:length(location_id)
    Pji[norm_col, :] ./= sum_over_destinations[norm_col]
end
maximum(abs.(sum(Pji, dims = 2) .- 1.0)) < 1e-14

total_trip_rate = 2 / 365
return_trip_rate = 1 / 5

phiji = total_trip_rate * Pji
tij = return_trip_rate

Nii = zeros(Float64, length(location_id))
for ni in 1:length(location_id)
    sumk = 0.0
    for nk in 1:length(location_id)
        sumk += phiji[nk, ni] / tij
    end
    Nii[ni] = N[ni] / (1 + sumk)
end
phiij = transpose(phiji)
Nij = similar(Pij)
for nj in 1:length(location_id)
    for ni in 1:length(location_id)
        Nij[ni, nj] = (phiij[ni, nj] / tij) * Nii[ni]
    end
end

tourist = zeros(Float64, length(location_id))
for tidx in 1:length(tourist)
    Nj = 0.0
    for i_idx in 1:length(tourist)
        Nj += Nij[i_idx, tidx]
    end
    tourist[tidx] = Nj / N[tidx]
end

sorted_tourist = sort(tourist)
using Plots
histogram(sorted_tourist)
savefig("tanzania_tourists.png")
