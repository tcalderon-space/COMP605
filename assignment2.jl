# Row vectors of A times the matrix B (update rows of `C`) with inner dot product
#2.1
function mygemm_ijp!(C, A, B)
  n, k = size(A)
  _, m = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (n, m)
  for i = 1:m
    for j = 1:n
      for p = 1:k
        @inbounds C[i, j] += A[i, p] * B[j, p]
      end
    end
  end
end

# Row vectors of A times the matrix B (update rows of `C`) with inner `axpy`
#2.2
function mygemm_ipj!(C, A, B)
    n, k = size(A)
    _, m = size(B)
    @assert size(B, 1) == k
    @assert size(C) == (n, m)

    for i = 1:m
      for p = 1:k
        for j = 1:n
          @inbounds C[i, j] += A[i, p] * B[j, p]
        end
      end
    end
end

# Rank one update (repeatedly update all elements of `C`) with outer product
# using `axpy` with columns of `A`
#2.4
function mygemm_pji!(C, A, B)
    n, k = size(A)
    _, m = size(B)
    @assert size(B, 1) == k
    @assert size(C) == (m, n)

    for p = 1:k
      for j = 1:n
        for i = 1:m
          @inbounds C[i, j] += A[i, p] * B[j, p]
        end
      end
    end
end

# matrix times column vector (update columns of `C`) with inner `axpy`
#2.5
function mygemm_jpi!(C, A, B)
    n, k = size(A)
    _, m = size(B)
    @assert size(B, 1) == k
    @assert size(C) == (m,n)

    for j = 1:n
      for p = 1:k
        for i = 1:m
          @inbounds C[i, j] += A[i, p] * B[j, p]
        end
      end
    end
end

# matrix times column vector (update columns of `C`) with inner dot product
2.6
function mygemm_jip!(C, A, B)
    n, k = size(A)
    _, m = size(B)
    @assert size(B, 1) == k
    @assert size(C) == (m,n)

    for j = 1:n
      for i = 1:m
        for p = 1:k
          @inbounds C[i, j] += A[i, p] * B[j, p]
        end
      end
    end
  end

# Rank one update (repeatedly update all elements of `C`) with outer product using `axpy` with rows of `B`
#Requested Assignment 2.3  loop ordering
function mygemm_pij!(C, A, B)
    m, k = size(A)
    _, n = size(B)
    @assert size(B, 1) == k
    @assert size(C) == (m, n)

    for p = 1:k
      for i = 1:m
        for j = 1:n
          @inbounds C[i, j] += A[i, p] * B[j, p]
        end
      end
    end
end

# What modules / packages do we depend on
using Random
using LinearAlgebra
using Printf
using Plots
default(linewidth=4) # Plots embelishments

# To ensure repeatability
Random.seed!(777)

# Don't let BLAS use lots of threads (since we are not multi-threaded yet!)
BLAS.set_num_threads(1)

# C := α * A * B' + β * C
refgemm!(C, A, B) = mul!(C, A, B', one(eltype(C)), one(eltype(C))) # to compare performance

# Algo 2.1: matrix times row vector (update rows of `C`) with inner dot product
#mygemm! = mygemm_ijp!

# Algo 2.2: matrix times row vector (update rows of `C`) with inner axpy
#mygemm! = mygemm_ipj!

# Algo 2.4: Rank one update (repeatedly update all elements of `C`) with outer product
# using axpy with columns of `A`
#mygemm! = mygemm_pji!

# Algo 2.5: matrix times column vector (update columns of `C`) with inner axpy || IGNORE
#mygemm! = mygemm_jpi!

# Algo 2.6: matrix times column vector (update columns of `C`) with inner dot product || IGNORE
#mygemm! = mygemm_jip!

# Requested algorithm
mygemm! = mygemm_pij!


num_reps = 3

# What precision numbers to use
# FloatType = Float32
FloatType = Float64

@printf("size |      reference      |           %s\n", mygemm!)
@printf("     |   seconds   GFLOPS  |   seconds   GFLOPS     err\n")

N = 48:48:480
best_perf = zeros(length(N))
# Size of square matrix to consider
for nmk in N
  i = Int(nmk / 48)
  n = m = k = nmk
  @printf("%4d |", nmk)

  gflops = 2 * m * n * k * 1e-09

  # Create the A, B, and C matrices with some random data
  A = rand(FloatType, m, k)
  B = rand(FloatType, k, n)
  C = rand(FloatType, m, n)

  # Make a copy of C for resetting data later
  C_old = copy(C)

  # The "truth", to check for correctness
  C_ref = C + A * B' # to check correctness, C_ref here will be your "truth"

  # Compute the reference timings
  best_time = typemax(FloatType)
  for iter = 1:num_reps
    # Reset C to the original data
    C .= C_old;
    run_time = @elapsed refgemm!(C, A, B);
    best_time = min(run_time, best_time)
  end
  # Make sure that we have the right answer!
  @assert C ≈ C_ref
  best_perf[i] = gflops / best_time

  # Print the reference implementation timing
  @printf("  %4.2e %8.2f  |", best_time, best_perf[i])

  # Compute the timing for mygemm! implementation
  best_time = typemax(FloatType)
  for iter = 1:num_reps
    # Reset C to the original data
    C .= C_old;
    run_time = @elapsed mygemm!(C, A, B);
    best_time = min(run_time, best_time)
  end
  best_perf[i] = gflops / best_time

  # Compute the error (difference between our implementation and the reference)
  err = norm(C - C_ref, Inf)

  # Print mygemm! implementations
  @printf("  %4.2e %8.2f   %.2e", best_time, best_perf[i], err)

  @printf("\n")
end

plot!(N, best_perf, xlabel = "m = n = k", ylabel = "GFLOPs/S", label = "$mygemm!", title = "GFLOPs/S w/ Matrices of m=n=k size with Float64")

num_reps = 3

# What precision numbers to use
FloatType = Float32
#FloatType = Float64

@printf("size |      reference      |           %s\n", mygemm!)
@printf("     |   seconds   GFLOPS  |   seconds   GFLOPS     err\n")

N = 48:48:480
best_perf = zeros(length(N))
# Size of square matrix to consider
for nmk in N
  i = Int(nmk / 48)
  n = m = k = nmk
  @printf("%4d |", nmk)

  gflops = 2 * m * n * k * 1e-09

  # Create the A, B, and C matrices with some random data
  A = rand(FloatType, m, k)
  B = rand(FloatType, k, n)
  C = rand(FloatType, m, n)

  # Make a copy of C for resetting data later
  C_old = copy(C)

  # The "truth", to check for correctness
  C_ref = C + A * B' # to check correctness, C_ref here will be your "truth"

  # Compute the reference timings
  best_time = typemax(FloatType)
  for iter = 1:num_reps
    # Reset C to the original data
    C .= C_old;
    run_time = @elapsed refgemm!(C, A, B);
    best_time = min(run_time, best_time)
  end
  # Make sure that we have the right answer!
  @assert C ≈ C_ref
  best_perf[i] = gflops / best_time

  # Print the reference implementation timing
  @printf("  %4.2e %8.2f  |", best_time, best_perf[i])

  # Compute the timing for mygemm! implementation
  best_time = typemax(FloatType)
  for iter = 1:num_reps
    # Reset C to the original data
    C .= C_old;
    run_time = @elapsed mygemm!(C, A, B);
    best_time = min(run_time, best_time)
  end
  best_perf[i] = gflops / best_time

  # Compute the error (difference between our implementation and the reference)
  err = norm(C - C_ref, Inf)

  # Print mygemm! implementations
  @printf("  %4.2e %8.2f   %.2e", best_time, best_perf[i], err)

  @printf("\n")
end

plot!(N, best_perf, xlabel = "m = n = k", ylabel = "GFLOPs/S", label = "$mygemm!", title = "GFLOPs/S vs. Matrices of m=n=k size w/Float64vsFloat32")

import Pkg;
Pkg.add("NBInclude")
Pkg.add("Random")
Pkg.add("LinearAlgebra")
Pkg.add("Printf")
Pkg.add("Plots")

using NBInclude
nbexport("assignment2.jl", "assignment2.ipynb")