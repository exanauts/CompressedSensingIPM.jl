# 1D
if dim1
  for N in (100, 200, 500)
    z = rand(N)

    z2_wei = M_perpt_M_perp_vec_wei(1, (N,), z, Int[])
    @test z2_wei ≈ z

    res1_wei = M_perpt_z_wei(1, (N,), z)
    @test norm(res1_wei) ≈ norm(z)

    res2_wei = M_perp_beta_wei(1, (N,), z, Int[])
    @test norm(res2_wei) ≈ norm(z)

    for rdft in (false, true)
      @testset "1D -- IPM -- CPU -- rdft=$rdft -- $N" begin
        nlp, solver, results, timer = ipm_example_1D(N; gpu=false, rdft, check=true)
        op_fft = nlp.op_fft

        z2 = M_perpt_M_perp_vec(op_fft, z)
        @test z2 ≈ z

        res1 = M_perpt_z(op_fft, z)
        @test norm(res1) ≈ norm(z)
        @test res1_wei ≈ res1

        res2 = M_perp_beta(op_fft, z)
        @test norm(res2) ≈ norm(z)
        @test res2_wei ≈ res2
      end

      if CUDA.functional()
        @testset "1D -- IPM -- CUDA -- rdft=$rdft -- $N" begin
          nlp, solver, results, timer = ipm_example_1D(N; gpu=true, gpu_arch="cuda", rdft, check=true)
          op_fft = nlp.op_fft

          z_gpu = CuArray(z)

          z2_gpu = M_perpt_M_perp_vec(op_fft, z_gpu)
          @test z2_gpu ≈ z_gpu

          res1_gpu = M_perpt_z(op_fft, z_gpu)
          @test norm(res1_gpu) ≈ norm(z_gpu)
          @test res1_wei ≈ collect(res1_gpu)

          res2_gpu = M_perp_beta(op_fft, z_gpu)
          @test norm(res2_gpu) ≈ norm(z_gpu)
          @test res2_wei ≈ collect(res2_gpu)
        end
      end

      if AMDGPU.functional()
        @testset "1D -- IPM -- ROCm -- rdft=$rdft -- $N" begin
          nlp, solver, results, timer = ipm_example_1D(N; gpu=true, gpu_arch="rocm", rdft, check=true)
          op_fft = nlp.op_fft

          z_gpu = ROCArray(z)

          z2_gpu = M_perpt_M_perp_vec(op_fft, z_gpu)
          @test z2_gpu ≈ z_gpu

          res1_gpu = M_perpt_z(op_fft, z_gpu)
          @test norm(res1_gpu) ≈ norm(z_gpu)
          @test res1_wei ≈ collect(res1_gpu)

          res2_gpu = M_perp_beta(op_fft, z_gpu)
          @test norm(res2_gpu) ≈ norm(z_gpu)
          @test res2_wei ≈ collect(res2_gpu)
        end
      end
    end
  end
end

# 2D
if dim2
  for (N1, N2) in ((16, 16), (12, 18), (16, 24), (32, 32))
    z = rand(N1 * N2)

    z2_wei = M_perpt_M_perp_vec_wei(2, (N1, N2), z, Int[])
    @test z2_wei ≈ z

    res1_wei = M_perpt_z_wei(2, (N1, N2), reshape(z, (N1, N2)))
    @test norm(res1_wei) ≈ norm(z)

    res2_wei = M_perp_beta_wei(2, (N1, N2), z, Int[])
    @test norm(res2_wei) ≈ norm(z)

    for rdft in (false, true)
      @testset "2D -- IPM -- CPU -- rdft=$rdft -- $N1 × $N2" begin
        nlp, solver, results, timer = ipm_example_2D(N1, N2; gpu=false, rdft, check=true)
        op_fft = nlp.op_fft

        z2 = M_perpt_M_perp_vec(op_fft, z)
        @test z2 ≈ z

        res1 = M_perpt_z(op_fft, reshape(z, (N1, N2)))
        @test norm(res1) ≈ norm(z)
        @test res1_wei ≈ res1

        res2 = M_perp_beta(op_fft, z)
        @test norm(res2) ≈ norm(z)
        @test res2_wei ≈ res2
      end

      if CUDA.functional()
        @testset "2D -- IPM -- CUDA -- rdft=$rdft -- $N1 × $N2" begin
          nlp, solver, results, timer = ipm_example_2D(N1, N2; gpu=true, gpu_arch="cuda", rdft, check=true)
          op_fft = nlp.op_fft

          z_gpu = CuArray(z)

          z2_gpu = M_perpt_M_perp_vec(op_fft, z_gpu)
          @test z2_gpu ≈ z_gpu

          res1_gpu = M_perpt_z(op_fft, reshape(z_gpu, (N1, N2)))
          @test norm(res1_gpu) ≈ norm(z_gpu)
          @test res1_wei ≈ collect(res1_gpu)

          res2_gpu = M_perp_beta(op_fft, z_gpu)
          @test norm(res2_gpu) ≈ norm(z_gpu)
          @test res2_wei ≈ collect(res2_gpu)
        end
      end
      if AMDGPU.functional()
        @testset "2D -- IPM -- ROCm -- rdft=$rdft -- $N1 × $N2" begin
          nlp, solver, results, timer = ipm_example_2D(N1, N2; gpu=true, gpu_arch="rocm", rdft, check=true)
          op_fft = nlp.op_fft

          z_gpu = ROCArray(z)

          z2_gpu = M_perpt_M_perp_vec(op_fft, z_gpu)
          @test z2_gpu ≈ z_gpu

          res1_gpu = M_perpt_z(op_fft, reshape(z_gpu, (N1, N2)))
          @test norm(res1_gpu) ≈ norm(z_gpu)
          @test res1_wei ≈ collect(res1_gpu)

          res2_gpu = M_perp_beta(op_fft, z_gpu)
          @test norm(res2_gpu) ≈ norm(z_gpu)
          @test res2_wei ≈ collect(res2_gpu)
        end
      end
    end
  end
end

# 3D
if dim3
  for (N1, N2, N3) in ((8, 8, 8), (2, 4, 6), (6, 10, 12), (14, 8, 4), (8, 6, 4), (16, 16, 16))
    z = rand(N1 * N2 * N3)

    z2_wei = M_perpt_M_perp_vec_wei(3, (N1, N2, N3), z, Int[])
    @test z2_wei ≈ z

    res1_wei = M_perpt_z_wei(3, (N1, N2, N3), reshape(z, (N1, N2, N3)))
    @test norm(res1_wei) ≈ norm(z)

    res2_wei = M_perp_beta_wei(3, (N1, N2, N3), z, Int[])
    @test norm(res2_wei) ≈ norm(z)

    for rdft in (false, true)
      @testset "3D -- IPM -- CPU -- rdft=$rdft -- $N1 × $N2 × $N3" begin
        nlp, solver, results, timer = ipm_example_3D(N1, N2, N3; gpu=false, rdft, check=true)
        op_fft = nlp.op_fft

        z2 = M_perpt_M_perp_vec(op_fft, z)
        @test z2 ≈ z

        res1 = M_perpt_z(op_fft, reshape(z, (N1, N2, N3)))
        @test norm(res1) ≈ norm(z)
        @test res1_wei ≈ res1

        res2 = M_perp_beta(op_fft, z)
        @test norm(res2) ≈ norm(z)
        @test res2_wei ≈ res2
      end

      if CUDA.functional()
        @testset "3D -- IPM -- CUDA -- rdft=$rdft -- $N1 × $N2 × $N3" begin
          nlp, solver, results, timer = ipm_example_3D(N1, N2, N3; gpu=true, gpu_arch="cuda", rdft, check=true)
          op_fft = nlp.op_fft

          z_gpu = CuArray(z)

          z2_gpu = M_perpt_M_perp_vec(op_fft, z_gpu)
          @test z2_gpu ≈ z_gpu

          res1_gpu = M_perpt_z(op_fft, reshape(z_gpu, (N1, N2, N3)))
          @test norm(res1_gpu) ≈ norm(z_gpu)
          @test res1_wei ≈ collect(res1_gpu)

          res2_gpu = M_perp_beta(op_fft, z_gpu)
          @test norm(res2_gpu) ≈ norm(z_gpu)
          @test res2_wei ≈ collect(res2_gpu)
        end
      end

      if AMDGPU.functional()
        @testset "3D -- IPM -- ROCm -- rdft=$rdft -- $N1 × $N2 × $N3" begin
          nlp, solver, results, timer = ipm_example_3D(N1, N2, N3; gpu=true, gpu_arch="rocm", rdft, check=true)
          op_fft = nlp.op_fft

          z_gpu = ROCArray(z)

          z2_gpu = M_perpt_M_perp_vec(op_fft, z_gpu)
          @test z2_gpu ≈ z_gpu

          res1_gpu = M_perpt_z(op_fft, reshape(z_gpu, (N1, N2, N3)))
          @test norm(res1_gpu) ≈ norm(z_gpu)
          @test res1_wei ≈ collect(res1_gpu)

          res2_gpu = M_perp_beta(op_fft, z_gpu)
          @test norm(res2_gpu) ≈ norm(z_gpu)
          @test res2_wei ≈ collect(res2_gpu)
        end
      end
    end
  end
end
