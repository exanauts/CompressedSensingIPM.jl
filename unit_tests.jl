using Test

# 1D
include("fft_example_1D.jl")
include("fft_example_2D.jl")
include("fft_example_3D.jl")

dim1 = false
dim2 = true
dim3 = false
cpu = true
gpu = true
rdft = false
for N in (100, 200, 500)
  if dim1 && cpu
    @testset "1D -- CPU -- $N" begin
      nlp, solver, results = fft_example_1D(N; gpu=false, rdft)

      z = rand(N)
      z2 = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (N,), z, Int[]; rdft)
      @test z2 ≈ z

      z2_wei = M_perpt_M_perp_vec_wei(1, (N,), z, Int[])
      @test z2_wei ≈ z

      res1 = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (N,), z; rdft)
      @test norm(res1) ≈ norm(z)

      res1_wei = M_perp_tz_wei(1, (N,), z)
      @test norm(res1_wei) ≈ norm(z)

      @test res1_wei ≈ res1

      res2 = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (N,), z, Int[]; rdft)
      @test norm(res2) ≈ norm(z)

      res2_wei = M_perp_beta_wei(1, (N,), z, Int[])
      @test norm(res2_wei) ≈ norm(z)

      @test res2_wei ≈ res2
    end
  end

  if dim1 && gpu && CUDA.functional()
    @testset "1D -- GPU -- $N" begin
      nlp, solver, results = fft_example_1D(N; gpu=true, rdft)

      z_gpu = CuArray(z)

      z2_gpu = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (N,), z_gpu, Int[]; rdft)
      @test z2_gpu ≈ z_gpu

      res1_gpu = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (N,), z_gpu; rdft)
      @test norm(res1_gpu) ≈ norm(z_gpu)

      @test res1_wei ≈ collect(res1_gpu)

      res2_gpu = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (N,), z_gpu, Int[]; rdft)
      @test norm(res2_gpu) ≈ norm(z_gpu)

      @test res2_wei ≈ collect(res2_gpu)
    end
  end
end

# 2D
for (N1, N2) in ((16, 16), (16, 24), (32, 32))
  if dim2 && cpu
    @testset "2D -- CPU -- $N1 × $N2" begin
      nlp, solver, results = fft_example_2D(N1, N2; gpu=false, rdft)

      z = rand(N1 * N2)
      z2 = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (N1, N2), z, Int[]; rdft)
      @test z2 ≈ z

      z2_wei = M_perpt_M_perp_vec_wei(2, (N1, N2), z, Int[])
      @test z2_wei ≈ z

      res1 = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (N1, N2), reshape(z, (N1, N2)); rdft)
      @test norm(res1) ≈ norm(z)

      res1_wei = M_perp_tz_wei(2, (N1, N2), reshape(z, (N1, N2)))
      @test norm(res1_wei) ≈ norm(z)

      @test res1_wei ≈ res1

      res2 = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (N1, N2), z, Int[]; rdft)
      @test norm(res2) ≈ norm(z)

      res2_wei = M_perp_beta_wei(2, (N1, N2), z, Int[])
      @test norm(res2_wei) ≈ norm(z)

      @test res2_wei ≈ res2
    end
  end

  if dim2 && gpu && CUDA.functional()
    @testset "2D -- GPU -- $N1 × $N2" begin
      nlp, solver, results = fft_example_2D(N1, N2; gpu=true, rdft)

      z_gpu = CuArray(z)

      z2_gpu = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (N1, N2), z_gpu, Int[]; rdft)
      @test z2_gpu ≈ z_gpu

      res1_gpu = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (N1, N2), reshape(z_gpu, (N1, N2)); rdft)
      @test norm(res1_gpu) ≈ norm(z_gpu)

      @test res1_wei ≈ collect(res1_gpu)

      res2_gpu = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (N1, N2), z_gpu, Int[]; rdft)
      @test norm(res2_gpu) ≈ norm(z_gpu)

      @test res2_wei ≈ collect(res2_gpu)
    end
  end
end

# 3D
for (N1, N2, N3) in ((8, 8, 8),)
  if dim3 && cpu
    @testset "3D -- CPU -- $N1 × $N2 × $N3" begin
      nlp, solver, results = fft_example_3D(N1, N2, N3; gpu=false, rdft)

      z = rand(N1 * N2 * N3)
      z2 = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (N1, N2, N3), z, Int[]; rdft)
      @test z2 ≈ z

      z2_wei = M_perpt_M_perp_vec_wei(3, (N1, N2, N3), z, Int[])
      @test z2_wei ≈ z

      res1 = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (N1, N2, N3), reshape(z, (N1, N2, N3)); rdft)
      @test norm(res1) ≈ norm(z)

      res1_wei = M_perp_tz_wei(3, (N1, N2, N3), reshape(z, (N1, N2, N3)))
      @test norm(res1_wei) ≈ norm(z)

      @test res1_wei ≈ res1

      res2 = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (N1, N2, N3), z, Int[]; rdft)
      @test norm(res2) ≈ norm(z)

      res2_wei = M_perp_beta_wei(3, (N1, N2, N3), z, Int[])
      @test norm(res2_wei) ≈ norm(z)

      @test res2_wei ≈ res2
    end
  end

  if dim3 && gpu && CUDA.functional()
    @testset "3D -- GPU -- $N1 × $N2 × $N3" begin
      nlp, solver, results = fft_example_3D(N1, N2, N3; gpu=true, rdft)

      z_gpu = CuArray(z)

      z2_gpu = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (N1, N2, N3), z_gpu, Int[]; rdft)
      @test z2_gpu ≈ z_gpu

      res1_gpu = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (N1, N2, N3), reshape(z_gpu, (N1, N2, N3)); rdft)
      @test norm(res1_gpu) ≈ norm(z_gpu)

      @test res1_wei ≈ collect(res1_gpu)

      res2_gpu = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (N1, N2, N3), z_gpu, Int[]; rdft)
      @test norm(res2_gpu) ≈ norm(z_gpu)

      @test res2_wei ≈ collect(res2_gpu)
    end
  end
end
