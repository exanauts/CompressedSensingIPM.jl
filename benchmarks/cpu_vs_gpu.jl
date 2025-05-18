using CompressedSensingIPM, FFTW
using MadNLP, MadNLPGPU
using CUDA, AMDGPU

include("../test/fft_wei.jl")
include("../test/punching_centering.jl")
include("../test/fft_example_3D.jl")

# CPU
gpu = false
rdft = true
check = false

for (N1, N2, N3) in [(8, 8, 8), (16, 16, 16),
                     (32, 32, 32), (64, 64, 64), (96, 96, 96),
                     (128, 128, 128), (192, 192, 192), (256, 256, 256),
                     (384, 384, 384), (512, 512, 512), (560, 560, 560)]

    println("$N1 | $N2 | $N3")
    nlp, solver, results, timer = fft_example_3D(N1, N2, N3; gpu, rdft, check)
    println("Timer: $(timer)")
end

# GPU
gpu = true
gpu_arch = "cuda"
rdft = true
check = false

for (N1, N2, N3) in [(8, 8, 8), (16, 16, 16),
                     (32, 32, 32), (64, 64, 64), (96, 96, 96),
                     (128, 128, 128), (192, 192, 192), (256, 256, 256),
                     (384, 384, 384), (512, 512, 512), (560, 560, 560)]

    println("$N1 | $N2 | $N3")
    nlp, solver, results, timer = fft_example_3D(N1, N2, N3; gpu, gpu_arch, rdft, check)
    println("Timer: $(timer)")
end
