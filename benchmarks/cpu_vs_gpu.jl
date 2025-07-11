using CompressedSensingIPM, FFTW
using MadNLP, MadNLPGPU
using CUDA, AMDGPU

include("../test/fft_wei.jl")
include("../test/punching_centering.jl")
include("../test/fft_example_3D.jl")

# CPU
run_cpu = true
rdft_cpu = true
check_cpu = false

if run_cpu
    for (N1, N2, N3) in [(8, 8, 8), (16, 16, 16),
                         (32, 32, 32), (64, 64, 64), (96, 96, 96),
                         (128, 128, 128), (192, 192, 192), (256, 256, 256),
                         (384, 384, 384), (512, 512, 512), (560, 560, 560)]

        println("$N1 | $N2 | $N3")
        nlp, solver, results, timer = fft_example_3D(N1, N2, N3; gpu=false, rdft=rdft_cpu, check=check_cpu)
        println("Timer: $(timer)")
    end
end

# GPU
run_gpu = true
rdft_gpu = true
check_gpu = false
gpu_arch = "cuda"  # "rocm"

if run_gpu
    for (N1, N2, N3) in [(8, 8, 8), (16, 16, 16),
                         (32, 32, 32), (64, 64, 64), (96, 96, 96),
                         (128, 128, 128), (192, 192, 192), (256, 256, 256),
                         (384, 384, 384), (512, 512, 512), (560, 560, 560)]

        println("$N1 | $N2 | $N3")
        nlp, solver, results, timer = fft_example_3D(N1, N2, N3; gpu=true, gpu_arch, rdft=rdft_gpu, check=check_gpu)
        println("Timer: $(timer)")
    end
end
