using CUDA, FFTW
p = 9
nrun_cpu = 10 * ones(Int, p)
nrun_cpu[p] = 1

nrun_gpu = 100 * ones(Int, p)

### fft ###
for i in 1:p
    n = Int(10^i)

    # CPU
    x = rand(ComplexF64, n)
    timer_cpu = Float64[]
    for k = 1:nrun_cpu[i]
        val = @elapsed fft(x)
        push!(timer_cpu, val)
    end
    timer_cpu = minimum(timer_cpu)
    timer_cpu_rounded = round(timer_cpu, digits=2)
    println("$i -- $n -- CPU -- $(timer_cpu)")
    println("$i -- $n -- CPU -- $(timer_cpu_rounded)")

    # GPU
    x = CuVector{ComplexF64}(x)
    timer_gpu = Float64[]
    for k = 1:nrun_gpu[i]
        val = CUDA.@elapsed CUDA.@sync fft(x)
        push!(timer_gpu, val)
    end
    timer_gpu = minimum(timer_gpu)
    timer_gpu_rounded = round(timer_gpu, digits=2)
    println("$i -- $n -- GPU -- $(timer_gpu)")
    println("$i -- $n -- GPU -- $(timer_gpu_rounded)")

    # RATIO
    ratio = timer_cpu / timer_gpu
    ratio_rounded = round(ratio, digits=2)
    println("$i -- $n -- RATIO -- $ratio")
    println("$i -- $n -- RATIO -- $ratio_rounded")
end

### ifft ###
for i in 1:p
    n = Int(10^i)

    # CPU
    x = rand(ComplexF64, n)
    timer_cpu = Float64[]
    for k = 1:nrun_cpu[i]
        val = @elapsed ifft(x)
        push!(timer_cpu, val)
    end
    timer_cpu = minimum(timer_cpu)
    timer_cpu_rounded = round(timer_cpu, digits=2)
    println("$i -- $n -- CPU -- $(timer_cpu)")
    println("$i -- $n -- CPU -- $(timer_cpu_rounded)")

    # GPU
    x = CuVector{ComplexF64}(x)
    timer_gpu = Float64[]
    for k = 1:nrun_gpu[i]
        val = CUDA.@elapsed CUDA.@sync ifft(x)
        push!(timer_gpu, val)
    end
    timer_gpu = minimum(timer_gpu)
    timer_gpu_rounded = round(timer_gpu, digits=2)
    println("$i -- $n -- GPU -- $(timer_gpu)")
    println("$i -- $n -- GPU -- $(timer_gpu_rounded)")

    # RATIO
    ratio = timer_cpu / timer_gpu
    ratio_rounded = round(ratio, digits=2)
    println("$i -- $n -- RATIO -- $ratio")
    println("$i -- $n -- RATIO -- $ratio_rounded")
end
