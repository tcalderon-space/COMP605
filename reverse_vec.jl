using CUDA
using CSV, DataFrames

# Loop-based reference CPU function the baseline
function reversevec!(a, b)
  N = length(a)
  for i = 1:N
    a[i] = b[N - i + 1]
  end
end

# Loop-based fake "GPU" kernel
#step 2
function fake_knl_reversevecs!(a, b, numblocks, bdim)
    N = length(a)
    # loop over the "blocks"
    @inbounds for bidx = 1:numblocks
        # loop over the "threads"
        for tidx = 1:bdim
        i = (bidx - 1) * bdim + tidx
        if i <= N
            a[i] = b[N - i + 1]
        end
        end
    end
end

# Real GPU kernel with an input vector b and an output vector a
function knl_reversevecs!(a, b)

    N = length(a)

    tidx = threadIdx().x # get the thread ID
    bidx = blockIdx().x  # get the block ID
    bdim = blockDim().x  # how many threads in each block

    # figure out which index we should handle
    i = (bidx - 1) * bdim + tidx
    @inbounds if i <= N
        a[i] = b[N - i + 1]
    end

    #Kernel must return nothing
    nothing
end

# Real GPU kernel for inplace reverse (a is both the input and output vector). 
#This version has a bug!
function knl_reversevecs_inplace_bad!(a)

    N = length(a)

    tidx = threadIdx().x # get the thread ID
    bidx = blockIdx().x  # get the block ID
    bdim = blockDim().x  # how many threads in each block


    # figure out which index we should handle
    i = (bidx - 1) * bdim + tidx
    @inbounds if i <= N
        a[i] = a[N - i + 1]
    end

    #Kernel must return nothing
    nothing
end

# Real GPU kernel for inplace reverse (a is both the input and output vector)
# real inplace rev GPU kernel
function knl_reversevecs_inplace!(a)

    ### TODO ###
    N = length(a)

    tidx = threadIdx().x # get the thread ID
    bidx = blockIdx().x  # get the block ID
    bdim = blockDim().x  # how many threads in each block

  

    # figure out which index we should handle
    i = (bidx - 1) * bdim + tidx

    @inbounds if i <= N % 2
      tmp = a[i]
      a[i] = a[N - i + 1]
      a[N - i + 1] = tmp
    end

    #Kernel must return nothing
    nothing

end

### Start of testing/driver code
using CUDA, DataFrames, CSV

let 
  N = 1000000000
  b = rand(Float32, N)
  a_ref = b[end:-1:1]

  #
  ### Simple reference for loop CPU implementation call
  #
  a1 = similar(a_ref)
  reversevec!(a1, b)  # Warm-up
  val, time, gctime, memory, _  = @timed reversevec!(a1, b)
  @assert a1 == a_ref

  df = DataFrame(
    time = [time],
    bytes = [memory]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  #
  ### Fake GPU kernel call
  #
  a2 = similar(a_ref)
  numthreads = 256
  numblocks = div(N + numthreads - 1, numthreads)
  fake_knl_reversevecs!(a2, b, numblocks, numthreads)  # Warm-up
  val, time, gctime, memory, _ = @timed fake_knl_reversevecs!(a2, b, numblocks, numthreads)
  @assert a_ref ≈ a2

  df = DataFrame(
    time = [time],
    bytes = [memory]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  #
  ### Double array reverse GPU kernel call
  #
  d_b = CuArray(b)
  d_a3 = CuArray{Float64}(undef, N)
  @cuda threads=numthreads blocks=numblocks knl_reversevecs!(d_a3, d_b)  # Warm-up
  synchronize()

  t = @timed begin
    @cuda threads=numthreads blocks=numblocks knl_reversevecs!(d_a3, d_b)
    synchronize()
  end

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  a3 = Array(d_a3)
  @assert a_ref ≈ a3

  #
  ### Inplace reverse GPU kernel (BAD) call
  #
  d_a4 = CuArray(b)
  @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace_bad!(d_a4)
  synchronize()
  d_a4 = CuArray(b)  # Reinit input
  t = @timed begin
    @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace_bad!(d_a4)
    synchronize()
  end

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  a4 = Array(d_a4)
  # @assert a_ref ≈ a4  # Left broken as per your comment

  #
  ### Inplace reverse GPU kernel (FIXED) call
  #
  d_a5 = CuArray(b)
  @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace!(d_a5)
  synchronize()
  d_a5 = CuArray(b)
  val, time, gctime, memory, allocations = @timed begin
    @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace!(d_a5)
    synchronize()
  end

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  a5 = Array(d_a5)


  #Float64
  b = rand(Float64, N)
  a_ref = b[end:-1:1]

  #
  ### Simple reference for loop CPU implementation call
  #
  a1 = similar(a_ref)
  reversevec!(a1, b)  # Warm-up
  val, time, gctime, memory, _  = @timed reversevec!(a1, b)
  @assert a1 == a_ref

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  #
  ### Fake GPU kernel call
  #
  a2 = similar(a_ref)
  numthreads = 256
  numblocks = div(N + numthreads - 1, numthreads)
  fake_knl_reversevecs!(a2, b, numblocks, numthreads)  # Warm-up
  val, time, gctime, memory, allocations = @timed fake_knl_reversevecs!(a2, b, numblocks, numthreads)
  @assert a_ref ≈ a2

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  #
  ### Double array reverse GPU kernel call
  #
  d_b = CuArray(b)
  d_a3 = CuArray{Float64}(undef, N)
  @cuda threads=numthreads blocks=numblocks knl_reversevecs!(d_a3, d_b)  # Warm-up
  synchronize()

  t = @timed begin
    @cuda threads=numthreads blocks=numblocks knl_reversevecs!(d_a3, d_b)
    synchronize()
  end

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  a3 = Array(d_a3)
  @assert a_ref ≈ a3

  #
  ### Inplace reverse GPU kernel (BAD) call
  #
  d_a4 = CuArray(b)
  @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace_bad!(d_a4)
  synchronize()
  d_a4 = CuArray(b)  # Reinit input
  t = @timed begin
    @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace_bad!(d_a4)
    synchronize()
  end

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  a4 = Array(d_a4)
  # @assert a_ref ≈ a4  # Left broken as per your comment

  #
  ### Inplace reverse GPU kernel (FIXED) call
  #
  d_a5 = CuArray(b)
  @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace!(d_a5)
  synchronize()
  d_a5 = CuArray(b)
  val, time, gctime, memory, allocations = @timed begin
    @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace!(d_a5)
    synchronize()
  end

  df = DataFrame(
    time = [t.time],
    bytes = [t.bytes]
  )
  CSV.write("reverse_vec_results3.csv", df, append=true)

  a5 = Array(d_a5)
  @assert a_ref ≈ a5
end
nothing
