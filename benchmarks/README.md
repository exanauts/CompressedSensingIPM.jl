## Benchmarks

This directory contains all the scripts and configuration files needed to reproduce the numerical results presented in the paper **Recovering Sparse DFT from Missing Signals via Interior Point Method on GPU**.

## Overview

Each script in this folder benchmarks different aspects of our GPU-accelerated interior-point solver and FFT implementations, comparing them against CPU-based references.

## Requirements

Ensure you have the appropriate hardware drivers installed (CUDA for NVIDIA GPUs, ROCm for AMD GPUs).

## Installation

1. Launch Julia with the project environment:
```shell
julia --project=.
```
2. Instantiate the environment:
```julia
using Pkg
Pkg.instantiate()
```

## Usage

To run a benchmark script, use one of the following commands:
```shell
julia --project=. -e 'include("benchmarks_cufft.jl")'
julia --project=. -e 'include("benchmarks_rocfft.jl")'
julia --project=. -e 'include("cpu_vs_gpu.jl")'
julia --project=. -e 'include("crystal.jl")'
julia --project=. -e 'include("mastodonte.jl")'
```

## Scripts

- **benchmarks_cufft.jl**  

Compares **cuFFT** (via CUDA.jl) against **FFTW** (via FFTW.jl) on problems of various sizes.
Measures execution time for `fft` and `ifft` operations on random data.

- **benchmarks_rocfft.jl**  

Compares **rocFFT** (via AMDGPU.jl) against **FFTW** (via FFTW.jl).
Similar to the cuFFT benchmarks; results were not included in the final paper.

- **cpu_vs_gpu.jl**  
 
 Benchmarks our compressed sensing solver on CPU vs GPU across a range of problem sizes (artificial test cases).

- **crystal.jl**  
 
Applies our compressed sensing solver on a real-world problem of **104 million variables**, comparing CPU and GPU performance on a crystallographic dataset.

- **mastodonte.jl** 

Applies our compressed sensing solver on problems from ANL stored in files `*.h5`.

## Preferences

To enable unified memory by default on the GH200, create a file named `LocalPreferences.toml` in this directory with the following content:

```toml
[CUDA]
default_memory = "unified"
```

## Acknowledgments

We thank [JLSE](https://www.jlse.anl.gov/) for providing access to the [NVIDIA GH200](https://www.jlse.anl.gov/nvidia-gh200) used in our experiments.
