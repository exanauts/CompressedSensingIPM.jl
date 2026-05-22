using AMDGPU, FFTW, BenchmarkTools, DelimitedFiles
import AMDGPU.rocFFT

include("fftw_guru.jl")

# One thread for reproducible CPU baseline (comment out to use all cores)
# FFTW.set_num_threads(1)

# Dimension: 1, 2, or 3  (pass as command-line argument)
#   julia benchmarks_rocfft.jl      → 1D  (default)
#   julia benchmarks_rocfft.jl 2    → 2D
#   julia benchmarks_rocfft.jl 3    → 3D
const DIM = isempty(ARGS) ? 1 : parse(Int, ARGS[1])
@assert DIM in (1, 2, 3) "DIM must be 1, 2, or 3"

# Problem sizes — side length n for square/cubic transforms
#   1D: n = 2^10 … 2^27  (total elements = n)
#   2D: n = 2^5  … 2^13  (total elements = n²,  up to 2^26)
#   3D: n = 2^3  … 2^9   (total elements = n³,  up to 2^27)
const sizes_1d = [Int(2^i) for i in 10:27]
const sizes_2d = [Int(2^i) for i in  5:13]
const sizes_3d = [Int(2^i) for i in  3: 9]

const sizes = DIM == 1 ? sizes_1d : DIM == 2 ? sizes_2d : sizes_3d

# ─── rocFFT plan wrapper ─────────────────────────────────────────────────────

mutable struct RocFFTPlan
    plan::rocFFT.rocfft_plan
    info::rocFFT.rocfft_execution_info
    work_buf::Union{ROCVector{UInt8}, Nothing}

    function RocFFTPlan(transform_type, dims::NTuple{D,Int};
                        in_type   = rocFFT.rocfft_array_type_complex_interleaved,
                        out_type  = rocFFT.rocfft_array_type_complex_interleaved,
                        placement = rocFFT.rocfft_placement_notinplace) where D
        desc_ref = Ref{rocFFT.rocfft_plan_description}()
        rocFFT.rocfft_plan_description_create(desc_ref)
        desc = desc_ref[]
        rocFFT.rocfft_plan_description_set_data_layout(
            desc, in_type, out_type,
            C_NULL, C_NULL,
            Csize_t(0), C_NULL, Csize_t(0),
            Csize_t(0), C_NULL, Csize_t(0),
        )
        plan_ref = Ref{rocFFT.rocfft_plan}()
        lengths = Csize_t[dims...]
        GC.@preserve lengths rocFFT.rocfft_plan_create(
            plan_ref, placement, transform_type,
            rocFFT.rocfft_precision_double,
            Csize_t(D), pointer(lengths), Csize_t(1),
            desc,
        )
        rocFFT.rocfft_plan_description_destroy(desc)
        plan = plan_ref[]

        info_ref = Ref{rocFFT.rocfft_execution_info}()
        rocFFT.rocfft_execution_info_create(info_ref)
        info = info_ref[]

        sz_ref = Ref{Csize_t}(0)
        rocFFT.rocfft_plan_get_work_buffer_size(plan, sz_ref)
        work_buf = nothing
        if sz_ref[] > 0
            work_buf = ROCVector{UInt8}(undef, Int(sz_ref[]))
            rocFFT.rocfft_execution_info_set_work_buffer(
                info, Ptr{Cvoid}(pointer(work_buf)), sz_ref[])
        end

        p = new(plan, info, work_buf)
        finalizer(p) do q
            rocFFT.rocfft_execution_info_destroy(q.info)
            rocFFT.rocfft_plan_destroy(q.plan)
            q.work_buf !== nothing && AMDGPU.unsafe_free!(q.work_buf)
        end
        return p
    end
end

# Backward-compatible 1D constructor
RocFFTPlan(transform_type, n::Int; kwargs...) =
    RocFFTPlan(transform_type, (n,); kwargs...)

function execute!(p::RocFFTPlan,
                  in_ptrs::Vector{Ptr{Cvoid}},
                  out_ptrs::Vector{Ptr{Cvoid}})
    GC.@preserve in_ptrs out_ptrs rocFFT.rocfft_execute(
        p.plan, pointer(in_ptrs), pointer(out_ptrs), p.info)
end

# ─── timing helpers ──────────────────────────────────────────────────────────

function gpu_belapsed(f; nwarmup::Int = 3, nsamples::Int = 10)
    for _ in 1:nwarmup
        f()
        AMDGPU.synchronize()
    end
    t_min = Inf
    for _ in 1:nsamples
        t = AMDGPU.@elapsed AMDGPU.@sync f()
        t < t_min && (t_min = t)
    end
    return t_min
end

# Adaptive sample counts based on total element count N = prod(dims):
#   N < 2^20  →  (nwarmup=3, nsamples=10)
#   2^20 ≤ N < 2^24  →  (2, 5)
#   N ≥ 2^24  →  (1, 3)
bench_params(dims::NTuple) = prod(dims) >= 2^24 ? (1, 3) :
                              prod(dims) >= 2^20 ? (2, 5) : (3, 10)
bench_params(n::Int) = bench_params((n,))

# ─── results collector ───────────────────────────────────────────────────────

const results = NamedTuple[]

function record!(transform, layout, ndim::Int, dims_str::String, n_total::Int,
                 execute_cpu, plan_cpu, execute_gpu, plan_gpu)
    tag = "$transform $(dims_str)"
    println("$tag -- EXECUTE_CPU   -- $execute_cpu")
    println("$tag -- PLAN_CPU      -- $plan_cpu")
    println("$tag -- EXECUTE_GPU   -- $execute_gpu")
    println("$tag -- PLAN_GPU      -- $plan_gpu")
    println("$tag -- EXECUTE_RATIO -- $(execute_cpu / execute_gpu)")
    println("$tag -- PLAN_RATIO    -- $(plan_cpu / plan_gpu)")
    push!(results, (
        transform     = transform,
        layout        = layout,
        ndim          = ndim,
        dims          = dims_str,
        n             = n_total,
        execute_cpu   = execute_cpu,
        plan_cpu      = plan_cpu,
        execute_gpu   = execute_gpu,
        plan_gpu      = plan_gpu,
        execute_ratio = execute_cpu / execute_gpu,
        plan_ratio    = plan_cpu / plan_gpu,
    ))
end

# Convenience wrapper for 1D (keeps all existing 1D call sites unchanged)
record!(transform, layout, n::Int, exe_cpu, plan_cpu, exe_gpu, plan_gpu) =
    record!(transform, layout, 1, string(n), n, exe_cpu, plan_cpu, exe_gpu, plan_gpu)

# ─── 1D Benchmarks ───────────────────────────────────────────────────────────

if DIM == 1

### C2C forward — interleaved ###
println("\n=== C2C forward (complex_interleaved) ===")
for n in sizes
    nw, ns = bench_params(n)
    x_cpu = rand(ComplexF64, n)
    y_cpu = Vector{ComplexF64}(undef, n)
    plan_cpu = plan_dft(n, x_cpu, y_cpu, _FFTW_FORWARD)
    execute!(plan_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_dft($n, $x_cpu, $y_cpu, _FFTW_FORWARD); finalize(p)) samples=ns evals=1

    x_gpu = ROCVector{ComplexF64}(x_cpu)
    y_gpu = ROCVector{ComplexF64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n); finalize(p)
    end

    record!("c2c_fwd", "interleaved", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### C2C inverse — interleaved ###
println("\n=== C2C inverse (complex_interleaved) ===")
for n in sizes
    nw, ns = bench_params(n)
    x_cpu = rand(ComplexF64, n)
    y_cpu = Vector{ComplexF64}(undef, n)
    plan_cpu = plan_dft(n, x_cpu, y_cpu, _FFTW_BACKWARD)
    execute!(plan_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_dft($n, $x_cpu, $y_cpu, _FFTW_BACKWARD); finalize(p)) samples=ns evals=1

    x_gpu = ROCVector{ComplexF64}(x_cpu)
    y_gpu = ROCVector{ComplexF64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n); finalize(p)
    end

    record!("c2c_inv", "interleaved", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### C2C forward — split/planar ###
println("\n=== C2C forward (split / complex_planar) ===")
for n in sizes
    nw, ns = bench_params(n)
    ri_cpu = rand(Float64, n);  ii_cpu = rand(Float64, n)
    ro_cpu = Vector{Float64}(undef, n);  io_cpu = Vector{Float64}(undef, n)
    plan_cpu = plan_split_dft(n, ri_cpu, ii_cpu, ro_cpu, io_cpu)
    execute!(plan_cpu, ri_cpu, ii_cpu, ro_cpu, io_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $ri_cpu, $ii_cpu, $ro_cpu, $io_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft($n, $ri_cpu, $ii_cpu, $ro_cpu, $io_cpu); finalize(p)) samples=ns evals=1

    ri_gpu = ROCVector{Float64}(ri_cpu);  ii_gpu = ROCVector{Float64}(ii_cpu)
    ro_gpu = ROCVector{Float64}(undef, n);  io_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n;
                          in_type  = rocFFT.rocfft_array_type_complex_planar,
                          out_type = rocFFT.rocfft_array_type_complex_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(ri_gpu)), Ptr{Cvoid}(pointer(ii_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(ro_gpu)), Ptr{Cvoid}(pointer(io_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n;
                       in_type  = rocFFT.rocfft_array_type_complex_planar,
                       out_type = rocFFT.rocfft_array_type_complex_planar); finalize(p)
    end

    record!("c2c_fwd", "split", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    for buf in (ri_gpu, ii_gpu, ro_gpu, io_gpu); AMDGPU.unsafe_free!(buf); end
end

### C2C inverse — split/planar ###
# CPU: forward plan executed with ri↔ii swapped (backward trick)
# GPU: complex_inverse + complex_planar
println("\n=== C2C inverse (split / complex_planar) ===")
for n in sizes
    nw, ns = bench_params(n)
    ri_cpu = rand(Float64, n);  ii_cpu = rand(Float64, n)
    ro_cpu = Vector{Float64}(undef, n);  io_cpu = Vector{Float64}(undef, n)
    plan_cpu = plan_split_dft(n, ii_cpu, ri_cpu, io_cpu, ro_cpu)
    execute!(plan_cpu, ii_cpu, ri_cpu, io_cpu, ro_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $ii_cpu, $ri_cpu, $io_cpu, $ro_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft($n, $ii_cpu, $ri_cpu, $io_cpu, $ro_cpu); finalize(p)) samples=ns evals=1

    ri_gpu = ROCVector{Float64}(ri_cpu);  ii_gpu = ROCVector{Float64}(ii_cpu)
    ro_gpu = ROCVector{Float64}(undef, n);  io_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n;
                          in_type  = rocFFT.rocfft_array_type_complex_planar,
                          out_type = rocFFT.rocfft_array_type_complex_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(ri_gpu)), Ptr{Cvoid}(pointer(ii_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(ro_gpu)), Ptr{Cvoid}(pointer(io_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n;
                       in_type  = rocFFT.rocfft_array_type_complex_planar,
                       out_type = rocFFT.rocfft_array_type_complex_planar); finalize(p)
    end

    record!("c2c_inv", "split", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    for buf in (ri_gpu, ii_gpu, ro_gpu, io_gpu); AMDGPU.unsafe_free!(buf); end
end

### R2C — interleaved ###
println("\n=== R2C (real → hermitian_interleaved) ===")
for n in sizes
    nw, ns = bench_params(n)
    x_cpu = rand(Float64, n)
    y_cpu = Vector{ComplexF64}(undef, n ÷ 2 + 1)
    plan_cpu = plan_dft_r2c(n, x_cpu, y_cpu)
    execute!(plan_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_dft_r2c($n, $x_cpu, $y_cpu); finalize(p)) samples=ns evals=1

    x_gpu = ROCVector{Float64}(x_cpu)
    y_gpu = ROCVector{ComplexF64}(undef, n ÷ 2 + 1)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                          in_type  = rocFFT.rocfft_array_type_real,
                          out_type = rocFFT.rocfft_array_type_hermitian_interleaved)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                       in_type  = rocFFT.rocfft_array_type_real,
                       out_type = rocFFT.rocfft_array_type_hermitian_interleaved); finalize(p)
    end

    record!("r2c", "interleaved", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### C2R — interleaved ###
println("\n=== C2R (hermitian_interleaved → real) ===")
for n in sizes
    nw, ns = bench_params(n)
    tmp = rand(Float64, n)
    ytmp = Vector{ComplexF64}(undef, n ÷ 2 + 1)
    p_init = plan_dft_r2c(n, tmp, ytmp); execute!(p_init, tmp, ytmp); finalize(p_init)
    x_cpu = ytmp
    y_cpu = Vector{Float64}(undef, n)
    plan_cpu = plan_dft_c2r(n, x_cpu, y_cpu)
    execute!(plan_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_dft_c2r($n, $x_cpu, $y_cpu); finalize(p)) samples=ns evals=1

    x_gpu = ROCVector{ComplexF64}(x_cpu)
    y_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                          in_type  = rocFFT.rocfft_array_type_hermitian_interleaved,
                          out_type = rocFFT.rocfft_array_type_real)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                       in_type  = rocFFT.rocfft_array_type_hermitian_interleaved,
                       out_type = rocFFT.rocfft_array_type_real); finalize(p)
    end

    record!("c2r", "interleaved", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### R2C — split/planar ###
println("\n=== R2C (real → hermitian_planar / split) ===")
for n in sizes
    nw, ns = bench_params(n)
    m = n ÷ 2 + 1
    x_cpu  = rand(Float64, n)
    ro_cpu = Vector{Float64}(undef, m)
    io_cpu = Vector{Float64}(undef, m)
    plan_cpu = plan_split_dft_r2c(n, x_cpu, ro_cpu, io_cpu)
    execute!(plan_cpu, x_cpu, ro_cpu, io_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $x_cpu, $ro_cpu, $io_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft_r2c($n, $x_cpu, $ro_cpu, $io_cpu); finalize(p)) samples=ns evals=1

    x_gpu  = ROCVector{Float64}(x_cpu)
    yr_gpu = ROCVector{Float64}(undef, m)
    yi_gpu = ROCVector{Float64}(undef, m)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                          in_type  = rocFFT.rocfft_array_type_real,
                          out_type = rocFFT.rocfft_array_type_hermitian_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(yr_gpu)), Ptr{Cvoid}(pointer(yi_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                       in_type  = rocFFT.rocfft_array_type_real,
                       out_type = rocFFT.rocfft_array_type_hermitian_planar); finalize(p)
    end

    record!("r2c", "split", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    for buf in (x_gpu, yr_gpu, yi_gpu); AMDGPU.unsafe_free!(buf); end
end

### C2R — split/planar ###
println("\n=== C2R (hermitian_planar / split → real) ===")
for n in sizes
    nw, ns = bench_params(n)
    m = n ÷ 2 + 1
    tmp = rand(Float64, n)
    ro_tmp = Vector{Float64}(undef, m); io_tmp = Vector{Float64}(undef, m)
    p_init = plan_split_dft_r2c(n, tmp, ro_tmp, io_tmp); execute!(p_init, tmp, ro_tmp, io_tmp); finalize(p_init)
    ri_cpu = ro_tmp;  ii_cpu = io_tmp
    out_cpu = Vector{Float64}(undef, n)
    plan_cpu = plan_split_dft_c2r(n, ri_cpu, ii_cpu, out_cpu)
    execute!(plan_cpu, ri_cpu, ii_cpu, out_cpu)
    t_exe_cpu  = @belapsed execute!($plan_cpu, $ri_cpu, $ii_cpu, $out_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft_c2r($n, $ri_cpu, $ii_cpu, $out_cpu); finalize(p)) samples=ns evals=1

    yr_gpu  = ROCVector{Float64}(ri_cpu)
    yi_gpu  = ROCVector{Float64}(ii_cpu)
    out_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                          in_type  = rocFFT.rocfft_array_type_hermitian_planar,
                          out_type = rocFFT.rocfft_array_type_real)
    in_ptrs  = [Ptr{Cvoid}(pointer(yr_gpu)), Ptr{Cvoid}(pointer(yi_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(out_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                       in_type  = rocFFT.rocfft_array_type_hermitian_planar,
                       out_type = rocFFT.rocfft_array_type_real); finalize(p)
    end

    record!("c2r", "split", n, t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(plan_cpu); finalize(plan_gpu)
    for buf in (yr_gpu, yi_gpu, out_gpu); AMDGPU.unsafe_free!(buf); end
end

end  # DIM == 1

# ─── 2D / 3D Benchmarks ──────────────────────────────────────────────────────
# CPU: FFTW high-level API (plan_fft / plan_bfft / plan_rfft / plan_irfft)
#      executed via FFTW.unsafe_execute! to avoid allocation during timing.
# GPU: RocFFTPlan with NTuple dims — interleaved layout only.
#      (split/planar for ND is uncommon and not benchmarked here.)

if DIM == 2 || DIM == 3

function _dims(n)
    DIM == 2 ? (n, n) : (n, n, n)
end
_dims_str(n) = DIM == 2 ? "$(n)x$(n)" : "$(n)x$(n)x$(n)"

### C2C forward — interleaved ###
println("\n=== $(DIM)D C2C forward (complex_interleaved) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)

    x_cpu = rand(ComplexF64, dims...)
    y_cpu = Array{ComplexF64}(undef, dims...)
    p_cpu = FFTW.plan_fft(x_cpu; flags=FFTW.ESTIMATE)
    FFTW.unsafe_execute!(p_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed FFTW.unsafe_execute!($p_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = FFTW.plan_fft($x_cpu; flags=FFTW.ESTIMATE); finalize(p)) samples=ns evals=1

    x_gpu = ROCArray(x_cpu)
    y_gpu = ROCArray{ComplexF64}(undef, dims...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, dims)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, dims); finalize(p)
    end

    record!("c2c_fwd", "interleaved", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### C2C inverse — interleaved ###
# CPU: plan_bfft = backward unnormalized FFT, equivalent to rocFFT complex_inverse
println("\n=== $(DIM)D C2C inverse (complex_interleaved) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)

    x_cpu = rand(ComplexF64, dims...)
    y_cpu = Array{ComplexF64}(undef, dims...)
    p_cpu = FFTW.plan_bfft(x_cpu; flags=FFTW.ESTIMATE)
    FFTW.unsafe_execute!(p_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed FFTW.unsafe_execute!($p_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = FFTW.plan_bfft($x_cpu; flags=FFTW.ESTIMATE); finalize(p)) samples=ns evals=1

    x_gpu = ROCArray(x_cpu)
    y_gpu = ROCArray{ComplexF64}(undef, dims...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, dims)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, dims); finalize(p)
    end

    record!("c2c_inv", "interleaved", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### R2C — interleaved ###
# Output hermitian shape: (n÷2+1, n[, n]) — first dim halved
println("\n=== $(DIM)D R2C (real → hermitian_interleaved) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)
    m = n ÷ 2 + 1
    dims_out = DIM == 2 ? (m, n) : (m, n, n)

    x_cpu = rand(Float64, dims...)
    y_cpu = Array{ComplexF64}(undef, dims_out...)
    p_cpu = FFTW.plan_rfft(x_cpu; flags=FFTW.ESTIMATE)
    FFTW.unsafe_execute!(p_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed FFTW.unsafe_execute!($p_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = FFTW.plan_rfft($x_cpu; flags=FFTW.ESTIMATE); finalize(p)) samples=ns evals=1

    x_gpu = ROCArray(x_cpu)
    y_gpu = ROCArray{ComplexF64}(undef, dims_out...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, dims;
                          in_type  = rocFFT.rocfft_array_type_real,
                          out_type = rocFFT.rocfft_array_type_hermitian_interleaved)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, dims;
                       in_type  = rocFFT.rocfft_array_type_real,
                       out_type = rocFFT.rocfft_array_type_hermitian_interleaved); finalize(p)
    end

    record!("r2c", "interleaved", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### C2R — interleaved ###
# Build a valid hermitian input via R2C, then benchmark C2R.
println("\n=== $(DIM)D C2R (hermitian_interleaved → real) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)
    m = n ÷ 2 + 1
    dims_out = DIM == 2 ? (m, n) : (m, n, n)

    tmp = rand(Float64, dims...)
    p_init = FFTW.plan_rfft(tmp; flags=FFTW.ESTIMATE)
    x_cpu  = Array{ComplexF64}(undef, dims_out...)
    FFTW.unsafe_execute!(p_init, tmp, x_cpu)
    finalize(p_init)
    y_cpu  = Array{Float64}(undef, dims...)
    p_cpu  = FFTW.plan_brfft(x_cpu, n; flags=FFTW.ESTIMATE)
    FFTW.unsafe_execute!(p_cpu, x_cpu, y_cpu)
    t_exe_cpu  = @belapsed FFTW.unsafe_execute!($p_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = FFTW.plan_brfft($x_cpu, $n; flags=FFTW.ESTIMATE); finalize(p)) samples=ns evals=1

    x_gpu = ROCArray(x_cpu)
    y_gpu = ROCArray{Float64}(undef, dims...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, dims;
                          in_type  = rocFFT.rocfft_array_type_hermitian_interleaved,
                          out_type = rocFFT.rocfft_array_type_real)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, dims;
                       in_type  = rocFFT.rocfft_array_type_hermitian_interleaved,
                       out_type = rocFFT.rocfft_array_type_real); finalize(p)
    end

    record!("c2r", "interleaved", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### C2C forward — split/planar ###
println("\n=== $(DIM)D C2C forward (split / complex_planar) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)

    ri_cpu = rand(Float64, dims...);  ii_cpu = rand(Float64, dims...)
    ro_cpu = Array{Float64}(undef, dims...);  io_cpu = Array{Float64}(undef, dims...)
    p_cpu = plan_split_dft_nd(dims, ri_cpu, ii_cpu, ro_cpu, io_cpu)
    execute!(p_cpu, ri_cpu, ii_cpu, ro_cpu, io_cpu)
    t_exe_cpu  = @belapsed execute!($p_cpu, $ri_cpu, $ii_cpu, $ro_cpu, $io_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft_nd($dims, $ri_cpu, $ii_cpu, $ro_cpu, $io_cpu); finalize(p)) samples=ns evals=1

    ri_gpu = ROCArray(ri_cpu);  ii_gpu = ROCArray(ii_cpu)
    ro_gpu = ROCArray{Float64}(undef, dims...);  io_gpu = ROCArray{Float64}(undef, dims...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, dims;
                          in_type  = rocFFT.rocfft_array_type_complex_planar,
                          out_type = rocFFT.rocfft_array_type_complex_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(ri_gpu)), Ptr{Cvoid}(pointer(ii_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(ro_gpu)), Ptr{Cvoid}(pointer(io_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, dims;
                       in_type  = rocFFT.rocfft_array_type_complex_planar,
                       out_type = rocFFT.rocfft_array_type_complex_planar); finalize(p)
    end

    record!("c2c_fwd", "split", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    for buf in (ri_gpu, ii_gpu, ro_gpu, io_gpu); AMDGPU.unsafe_free!(buf); end
end

### C2C inverse — split/planar ###
# CPU: forward plan executed with ri↔ii swapped (backward trick)
# GPU: complex_inverse + complex_planar
println("\n=== $(DIM)D C2C inverse (split / complex_planar) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)

    ri_cpu = rand(Float64, dims...);  ii_cpu = rand(Float64, dims...)
    ro_cpu = Array{Float64}(undef, dims...);  io_cpu = Array{Float64}(undef, dims...)
    p_cpu = plan_split_dft_nd(dims, ii_cpu, ri_cpu, io_cpu, ro_cpu)
    execute!(p_cpu, ii_cpu, ri_cpu, io_cpu, ro_cpu)
    t_exe_cpu  = @belapsed execute!($p_cpu, $ii_cpu, $ri_cpu, $io_cpu, $ro_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft_nd($dims, $ii_cpu, $ri_cpu, $io_cpu, $ro_cpu); finalize(p)) samples=ns evals=1

    ri_gpu = ROCArray(ri_cpu);  ii_gpu = ROCArray(ii_cpu)
    ro_gpu = ROCArray{Float64}(undef, dims...);  io_gpu = ROCArray{Float64}(undef, dims...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, dims;
                          in_type  = rocFFT.rocfft_array_type_complex_planar,
                          out_type = rocFFT.rocfft_array_type_complex_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(ri_gpu)), Ptr{Cvoid}(pointer(ii_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(ro_gpu)), Ptr{Cvoid}(pointer(io_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, dims;
                       in_type  = rocFFT.rocfft_array_type_complex_planar,
                       out_type = rocFFT.rocfft_array_type_complex_planar); finalize(p)
    end

    record!("c2c_inv", "split", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    for buf in (ri_gpu, ii_gpu, ro_gpu, io_gpu); AMDGPU.unsafe_free!(buf); end
end

### R2C — split/planar ###
println("\n=== $(DIM)D R2C (real → hermitian_planar / split) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)
    m = n ÷ 2 + 1
    dims_out = DIM == 2 ? (m, n) : (m, n, n)

    x_cpu  = rand(Float64, dims...)
    ro_cpu = Array{Float64}(undef, dims_out...)
    io_cpu = Array{Float64}(undef, dims_out...)
    p_cpu = plan_split_dft_r2c_nd(dims, x_cpu, ro_cpu, io_cpu)
    execute!(p_cpu, x_cpu, ro_cpu, io_cpu)
    t_exe_cpu  = @belapsed execute!($p_cpu, $x_cpu, $ro_cpu, $io_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft_r2c_nd($dims, $x_cpu, $ro_cpu, $io_cpu); finalize(p)) samples=ns evals=1

    x_gpu  = ROCArray(x_cpu)
    yr_gpu = ROCArray{Float64}(undef, dims_out...)
    yi_gpu = ROCArray{Float64}(undef, dims_out...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, dims;
                          in_type  = rocFFT.rocfft_array_type_real,
                          out_type = rocFFT.rocfft_array_type_hermitian_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(yr_gpu)), Ptr{Cvoid}(pointer(yi_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, dims;
                       in_type  = rocFFT.rocfft_array_type_real,
                       out_type = rocFFT.rocfft_array_type_hermitian_planar); finalize(p)
    end

    record!("r2c", "split", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    for buf in (x_gpu, yr_gpu, yi_gpu); AMDGPU.unsafe_free!(buf); end
end

### C2R — split/planar ###
println("\n=== $(DIM)D C2R (hermitian_planar / split → real) ===")
for n in sizes
    dims = _dims(n)
    nw, ns = bench_params(dims)
    dims_str = _dims_str(n)
    n_total = prod(dims)
    m = n ÷ 2 + 1
    dims_out = DIM == 2 ? (m, n) : (m, n, n)

    tmp    = rand(Float64, dims...)
    ro_tmp = Array{Float64}(undef, dims_out...);  io_tmp = Array{Float64}(undef, dims_out...)
    p_init = plan_split_dft_r2c_nd(dims, tmp, ro_tmp, io_tmp)
    execute!(p_init, tmp, ro_tmp, io_tmp);  finalize(p_init)
    ri_cpu = ro_tmp;  ii_cpu = io_tmp
    out_cpu = Array{Float64}(undef, dims...)
    p_cpu = plan_split_dft_c2r_nd(dims, ri_cpu, ii_cpu, out_cpu)
    execute!(p_cpu, ri_cpu, ii_cpu, out_cpu)
    t_exe_cpu  = @belapsed execute!($p_cpu, $ri_cpu, $ii_cpu, $out_cpu) samples=ns evals=1
    t_plan_cpu = @belapsed (p = plan_split_dft_c2r_nd($dims, $ri_cpu, $ii_cpu, $out_cpu); finalize(p)) samples=ns evals=1

    yr_gpu  = ROCArray(ri_cpu)
    yi_gpu  = ROCArray(ii_cpu)
    out_gpu = ROCArray{Float64}(undef, dims...)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, dims;
                          in_type  = rocFFT.rocfft_array_type_hermitian_planar,
                          out_type = rocFFT.rocfft_array_type_real)
    in_ptrs  = [Ptr{Cvoid}(pointer(yr_gpu)), Ptr{Cvoid}(pointer(yi_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(out_gpu))]
    t_exe_gpu  = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    t_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, dims;
                       in_type  = rocFFT.rocfft_array_type_hermitian_planar,
                       out_type = rocFFT.rocfft_array_type_real); finalize(p)
    end

    record!("c2r", "split", DIM, dims_str, n_total,
            t_exe_cpu, t_plan_cpu, t_exe_gpu, t_plan_gpu)
    finalize(p_cpu); finalize(plan_gpu)
    for buf in (yr_gpu, yi_gpu, out_gpu); AMDGPU.unsafe_free!(buf); end
end

end  # DIM == 2 || DIM == 3

# ─── Save results ─────────────────────────────────────────────────────────────

println("\nSaving results...")

outfile = "results_rocfft_$(DIM)d.csv"
header = ["transform" "layout" "ndim" "dims" "n" "execute_cpu" "plan_cpu" "execute_gpu" "plan_gpu" "execute_ratio" "plan_ratio"]
data = hcat(
    [r.transform     for r in results],
    [r.layout        for r in results],
    [r.ndim          for r in results],
    [r.dims          for r in results],
    [r.n             for r in results],
    [r.execute_cpu   for r in results],
    [r.plan_cpu      for r in results],
    [r.execute_gpu   for r in results],
    [r.plan_gpu      for r in results],
    [r.execute_ratio for r in results],
    [r.plan_ratio    for r in results],
)
writedlm(outfile, vcat(header, data), ',')
println("Results saved to $outfile ($(length(results)) rows)")
