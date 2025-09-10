# 1D
if dim1
  for N in (100, 200, 500)
    for rdft in (false, true)
      @testset "1D -- ADMM -- CPU -- rdft=$rdft -- $N" begin
        solution, timer = admm_example_1D(N; gpu=false, rdft, check=true)
      end

      # if CUDA.functional()
      #   @testset "1D -- ADMM -- CUDA -- rdft=$rdft -- $N" begin
      #     solution, timer = admm_example_1D(N; gpu=true, gpu_arch="cuda", rdft, check=true)
      #   end
      # end

      # if AMDGPU.functional()
      #   @testset "1D -- ADMM -- ROCm -- rdft=$rdft -- $N" begin
      #     solution, timer = admm_example_1D(N; gpu=true, gpu_arch="rocm", rdft, check=true)
      #   end
      # end
    end
  end
end


# 2D
if dim2
  for (N1, N2) in ((16, 16), (12, 18), (16, 24), (32, 32))
    for rdft in (false, true)
      @testset "2D -- ADMM -- CPU -- rdft=$rdft -- $N1 × $N2" begin
        solution, timer = admm_example_2D(N1, N2; gpu=false, rdft, check=true)
      end

      # if CUDA.functional()
      #   @testset "2D -- ADMM -- CUDA -- rdft=$rdft -- $N1 × $N2" begin
      #     solution, timer = admm_example_2D(N1, N2; gpu=true, gpu_arch="cuda", rdft, check=true)
      #   end
      # end

      # if AMDGPU.functional()
      #   @testset "2D -- ADMM -- ROCm -- rdft=$rdft -- $N1 × $N2" begin
      #     solution, timer = admm_example_2D(N1, N2; gpu=true, gpu_arch="rocm", rdft, check=true)
      #   end
      # end
    end
  end
end


# 3D
if dim3
  for (N1, N2, N3) in ((8, 8, 8), (2, 4, 6), (6, 10, 12), (14, 8, 4), (8, 6, 4), (16, 16, 16))
    for rdft in (false, true)
      @testset "3D -- ADMM -- CPU -- rdft=$rdft -- $N1 × $N2 × $N3" begin
        solution, timer = admm_example_3D(N1, N2, N3; gpu=false, rdft, check=true)
      end

      # if CUDA.functional()
      #   @testset "3D -- ADMM -- CUDA -- rdft=$rdft -- $N1 × $N2 × $N3" begin
      #     solution, timer = admm_example_3D(N1, N2, N3; gpu=true, gpu_arch="cuda", rdft, check=true)
      #   end
      # end

      # if AMDGPU.functional()
      #   @testset "3D -- ADMM -- ROCm -- rdft=$rdft -- $N1 × $N2 × $N3" begin
      #     solution, timer = admm_example_3D(N1, N2, N3; gpu=true, gpu_arch="rocm", rdft, check=true)
      #   end
      # end
    end
  end
end
