using AMDGPU, FFTW, BenchmarkTools
p = 8

### fft ###
for i in 1:p
    n = Int(10^i)
    x = rand(Float64, n)
    fft(x)  # warm up
    timer_cpu = @belapsed fft($x)
    println("$n -- CPU -- $(timer_cpu)")
    x = ROCVector{Float64}(x)
    fft(x)  # warm up
    timer_gpu = AMDGPU.@elapsed AMDGPU.@sync fft(x)
    println("$n -- GPU -- $(timer_gpu)")
    ratio = timer_cpu / timer_gpu
    println("$n -- RATIO -- $ratio")
    AMDGPU.unsafe_free!(x)
end

### ifft ###
for i in 1:p
    n = Int(10^i)
    x = rand(ComplexF64, n)
    ifft(x)  # warm up
    timer_cpu = @belapsed ifft($x)
    println("$n -- CPU -- $(timer_cpu)")
    x = ROCVector{ComplexF64}(x)
    ifft(x)  # warm up
    timer_gpu = AMDGPU.@elapsed AMDGPU.@sync ifft(x)
    println("$n -- GPU -- $(timer_gpu)")
    ratio = timer_cpu / timer_gpu
    println("$n -- RATIO -- $ratio")
    AMDGPU.unsafe_free!(x)
end
