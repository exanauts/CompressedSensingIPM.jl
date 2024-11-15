using Test

# 1D
include("fft_example_1D.jl")

z = rand(100)
z2 = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (100,), z, Int[])
@test z2 ≈ z

z2_wei = M_perpt_M_perp_vec_wei(1, (100,), z, Int[])
@test z2_wei ≈ z

res1 = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (100,), z)
@test norm(res1) ≈ norm(z)

res1_wei = M_perp_tz_wei(1, (100,), z)
@test norm(res1_wei) ≈ norm(z)

@test res1_wei ≈ res1

res2 = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (100,), z, Int[])
@test norm(res2) ≈ norm(z)

res2_wei = M_perp_beta_wei(1, (100,), z, Int[])
@test norm(res2_wei) ≈ norm(z)

@test res2_wei ≈ res2

if CUDA.functional()
  include("fft_example_1D_gpu.jl")
  z_gpu = CuArray(z)

  z2_gpu = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (100,), z_gpu, Int[])
  @test z2_gpu ≈ z_gpu

  res1_gpu = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (100,), z_gpu)
  @test norm(res1_gpu) ≈ norm(z_gpu)

  @test res1_wei ≈ collect(res1_gpu)

  res2_gpu = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 1, (100,), z_gpu, Int[])
  @test norm(res2_gpu) ≈ norm(z_gpu)

  @test res2_wei ≈ collect(res2_gpu)
end

# 2D
include("fft_example_2D.jl")

z = rand(16 * 16)
z2 = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (16, 16), z, Int[])
@test z2 ≈ z

z2_wei = M_perpt_M_perp_vec_wei(2, (16, 16), z, Int[])
@test z2_wei ≈ z

res1 = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (16, 16), reshape(z, (16, 16)))
@test norm(res1) ≈ norm(z)

res1_wei = M_perp_tz_wei(2, (16, 16), reshape(z, (16, 16)))
@test norm(res1_wei) ≈ norm(z)

@test res1_wei ≈ res1

res2 = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 2, (16, 16), z, Int[])
@test norm(res2) ≈ norm(z)

res2_wei = M_perp_beta_wei(2, (16, 16), z, Int[])
@test norm(res2_wei) ≈ norm(z)

@test res2_wei ≈ res2

if CUDA.functional()
  include("fft_example_2D_gpu.jl")
  z_gpu = CuArray(z)

  z2_gpu = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op,  2, (16, 16), z_gpu, Int[])
  @test z2_gpu ≈ z_gpu

  res1_gpu = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op,  2, (16, 16), reshape(z_gpu, (16, 16)))
  @test norm(res1_gpu) ≈ norm(z_gpu)

  @test res1_wei ≈ collect(res1_gpu)

  res2_gpu = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op,  2, (16, 16), z_gpu, Int[])
  @test norm(res2_gpu) ≈ norm(z_gpu)

  @test res2_wei ≈ collect(res2_gpu)
end

# 3D
include("fft_example_3D.jl")

z = rand(8 * 8 * 8)
z2 = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (8, 8, 8), z, Int[])
@test z2 ≈ z

z2_wei = M_perpt_M_perp_vec_wei(3, (8, 8, 8), z, Int[])
@test z2_wei ≈ z

res1 = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (8, 8, 8), reshape(z, (8, 8, 8)))
@test norm(res1) ≈ norm(z)

res1_wei = M_perp_tz_wei(3, (8, 8, 8), reshape(z, (8, 8, 8)))
@test norm(res1_wei) ≈ norm(z)

@test res1_wei ≈ res1

res2 = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (8, 8, 8), z, Int[])
@test norm(res2) ≈ norm(z)

res2_wei = M_perp_beta_wei(3, (8, 8, 8), z, Int[])
@test norm(res2_wei) ≈ norm(z)

@test res2_wei ≈ res2

# if CUDA.functional()
#   include("fft_example_3D_gpu.jl")
#   z_gpu = CuArray(z)

#   z2_gpu = M_perpt_M_perp_vec(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (8, 8, 8), z_gpu, Int[])
#   @test z2_gpu ≈ z_gpu

#   res1_gpu = M_perp_tz(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (8, 8, 8), reshape(z_gpu, (8, 8, 8)))
#   @test norm(res1_gpu) ≈ norm(z_gpu)

#   @test res1_wei ≈ collect(res1_gpu)

#   res2_gpu = M_perp_beta(nlp.buffer_real, nlp.buffer_complex1, nlp.buffer_complex2, nlp.op, 3, (8, 8, 8), z_gpu, Int[])
#   @test norm(res2_gpu) ≈ norm(z_gpu)

#   @test res2_wei ≈ collect(res2_gpu)
# end
