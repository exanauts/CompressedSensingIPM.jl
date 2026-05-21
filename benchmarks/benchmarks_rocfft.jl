using AMDGPU, FFTW, BenchmarkTools
import AMDGPU.rocFFT

include("fftw_guru.jl")

# One thread for reproducible CPU baseline (comment out to use all cores)
# FFTW.set_num_threads(1)

const sizes = [Int(2^i) for i in 10:27]

# ─── rocFFT plan wrapper ─────────────────────────────────────────────────────

mutable struct RocFFTPlan
    plan::rocFFT.rocfft_plan
    info::rocFFT.rocfft_execution_info
    work_buf::Union{ROCVector{UInt8}, Nothing}

    function RocFFTPlan(transform_type, n;
                        in_type   = rocFFT.rocfft_array_type_complex_interleaved,
                        out_type  = rocFFT.rocfft_array_type_complex_interleaved,
                        placement = rocFFT.rocfft_placement_notinplace)
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
        lengths = Csize_t[n]
        GC.@preserve lengths rocFFT.rocfft_plan_create(
            plan_ref, placement, transform_type,
            rocFFT.rocfft_precision_double,
            Csize_t(1), pointer(lengths), Csize_t(1),
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

# Reduce sample counts for large n to keep total runtime reasonable:
#   n < 2^20  →  (nwarmup=3, nsamples=10)
#   2^20 ≤ n < 2^24  →  (2, 5)
#   n ≥ 2^24  →  (1, 3)
function bench_params(n::Int)
    n >= 2^24 && return (1, 3)
    n >= 2^20 && return (2, 5)
    return (3, 10)
end

# ─── Benchmarks ──────────────────────────────────────────────────────────────

### C2C forward — interleaved ###
println("\n=== C2C forward (complex_interleaved) ===")
for n in sizes
    nw, ns = bench_params(n)
    x_cpu = rand(ComplexF64, n)
    y_cpu = Vector{ComplexF64}(undef, n)
    plan_cpu = plan_dft(n, x_cpu, y_cpu, _FFTW_FORWARD)
    execute!(plan_cpu, x_cpu, y_cpu)
    timer_cpu      = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_dft($n, $x_cpu, $y_cpu, _FFTW_FORWARD); finalize(p)) samples=ns evals=1
    println("c2c_fwd $n -- CPU      -- $timer_cpu")
    println("c2c_fwd $n -- PLAN_CPU -- $timer_plan_cpu")

    x_gpu = ROCVector{ComplexF64}(x_cpu)
    y_gpu = ROCVector{ComplexF64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n)
        finalize(p)
    end
    println("c2c_fwd $n -- GPU      -- $timer_gpu")
    println("c2c_fwd $n -- PLAN_GPU -- $timer_plan_gpu")
    println("c2c_fwd $n -- RATIO    -- $(timer_cpu / timer_gpu)")

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
    timer_cpu      = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_dft($n, $x_cpu, $y_cpu, _FFTW_BACKWARD); finalize(p)) samples=ns evals=1
    println("c2c_inv $n -- CPU      -- $timer_cpu")
    println("c2c_inv $n -- PLAN_CPU -- $timer_plan_cpu")

    x_gpu = ROCVector{ComplexF64}(x_cpu)
    y_gpu = ROCVector{ComplexF64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n)
        finalize(p)
    end
    println("c2c_inv $n -- GPU      -- $timer_gpu")
    println("c2c_inv $n -- PLAN_GPU -- $timer_plan_gpu")
    println("c2c_inv $n -- RATIO    -- $(timer_cpu / timer_gpu)")

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
    timer_cpu      = @belapsed execute!($plan_cpu, $ri_cpu, $ii_cpu, $ro_cpu, $io_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_split_dft($n, $ri_cpu, $ii_cpu, $ro_cpu, $io_cpu); finalize(p)) samples=ns evals=1
    println("c2c_fwd_split $n -- CPU      -- $timer_cpu")
    println("c2c_fwd_split $n -- PLAN_CPU -- $timer_plan_cpu")

    ri_gpu = ROCVector{Float64}(ri_cpu);  ii_gpu = ROCVector{Float64}(ii_cpu)
    ro_gpu = ROCVector{Float64}(undef, n);  io_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n;
                          in_type  = rocFFT.rocfft_array_type_complex_planar,
                          out_type = rocFFT.rocfft_array_type_complex_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(ri_gpu)), Ptr{Cvoid}(pointer(ii_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(ro_gpu)), Ptr{Cvoid}(pointer(io_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_forward, n;
                       in_type  = rocFFT.rocfft_array_type_complex_planar,
                       out_type = rocFFT.rocfft_array_type_complex_planar)
        finalize(p)
    end
    println("c2c_fwd_split $n -- GPU      -- $timer_gpu")
    println("c2c_fwd_split $n -- PLAN_GPU -- $timer_plan_gpu")
    println("c2c_fwd_split $n -- RATIO    -- $(timer_cpu / timer_gpu)")

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
    timer_cpu      = @belapsed execute!($plan_cpu, $ii_cpu, $ri_cpu, $io_cpu, $ro_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_split_dft($n, $ii_cpu, $ri_cpu, $io_cpu, $ro_cpu); finalize(p)) samples=ns evals=1
    println("c2c_inv_split $n -- CPU      -- $timer_cpu")
    println("c2c_inv_split $n -- PLAN_CPU -- $timer_plan_cpu")

    ri_gpu = ROCVector{Float64}(ri_cpu);  ii_gpu = ROCVector{Float64}(ii_cpu)
    ro_gpu = ROCVector{Float64}(undef, n);  io_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n;
                          in_type  = rocFFT.rocfft_array_type_complex_planar,
                          out_type = rocFFT.rocfft_array_type_complex_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(ri_gpu)), Ptr{Cvoid}(pointer(ii_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(ro_gpu)), Ptr{Cvoid}(pointer(io_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_complex_inverse, n;
                       in_type  = rocFFT.rocfft_array_type_complex_planar,
                       out_type = rocFFT.rocfft_array_type_complex_planar)
        finalize(p)
    end
    println("c2c_inv_split $n -- GPU      -- $timer_gpu")
    println("c2c_inv_split $n -- PLAN_GPU -- $timer_plan_gpu")
    println("c2c_inv_split $n -- RATIO    -- $(timer_cpu / timer_gpu)")

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
    timer_cpu      = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_dft_r2c($n, $x_cpu, $y_cpu); finalize(p)) samples=ns evals=1
    println("r2c_interleaved $n -- CPU      -- $timer_cpu")
    println("r2c_interleaved $n -- PLAN_CPU -- $timer_plan_cpu")

    x_gpu = ROCVector{Float64}(x_cpu)
    y_gpu = ROCVector{ComplexF64}(undef, n ÷ 2 + 1)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                          in_type  = rocFFT.rocfft_array_type_real,
                          out_type = rocFFT.rocfft_array_type_hermitian_interleaved)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                       in_type  = rocFFT.rocfft_array_type_real,
                       out_type = rocFFT.rocfft_array_type_hermitian_interleaved)
        finalize(p)
    end
    println("r2c_interleaved $n -- GPU      -- $timer_gpu")
    println("r2c_interleaved $n -- PLAN_GPU -- $timer_plan_gpu")
    println("r2c_interleaved $n -- RATIO    -- $(timer_cpu / timer_gpu)")

    finalize(plan_cpu); finalize(plan_gpu)
    AMDGPU.unsafe_free!(x_gpu); AMDGPU.unsafe_free!(y_gpu)
end

### C2R — interleaved ###
println("\n=== C2R (hermitian_interleaved → real) ===")
for n in sizes
    nw, ns = bench_params(n)
    # build a valid hermitian spectrum from a real signal
    tmp = rand(Float64, n)
    ytmp = Vector{ComplexF64}(undef, n ÷ 2 + 1)
    p_init = plan_dft_r2c(n, tmp, ytmp); execute!(p_init, tmp, ytmp); finalize(p_init)
    x_cpu = ytmp
    y_cpu = Vector{Float64}(undef, n)
    plan_cpu = plan_dft_c2r(n, x_cpu, y_cpu)
    execute!(plan_cpu, x_cpu, y_cpu)
    timer_cpu      = @belapsed execute!($plan_cpu, $x_cpu, $y_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_dft_c2r($n, $x_cpu, $y_cpu); finalize(p)) samples=ns evals=1
    println("c2r_interleaved $n -- CPU      -- $timer_cpu")
    println("c2r_interleaved $n -- PLAN_CPU -- $timer_plan_cpu")

    x_gpu = ROCVector{ComplexF64}(x_cpu)
    y_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                          in_type  = rocFFT.rocfft_array_type_hermitian_interleaved,
                          out_type = rocFFT.rocfft_array_type_real)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(y_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                       in_type  = rocFFT.rocfft_array_type_hermitian_interleaved,
                       out_type = rocFFT.rocfft_array_type_real)
        finalize(p)
    end
    println("c2r_interleaved $n -- GPU      -- $timer_gpu")
    println("c2r_interleaved $n -- PLAN_GPU -- $timer_plan_gpu")
    println("c2r_interleaved $n -- RATIO    -- $(timer_cpu / timer_gpu)")

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
    timer_cpu      = @belapsed execute!($plan_cpu, $x_cpu, $ro_cpu, $io_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_split_dft_r2c($n, $x_cpu, $ro_cpu, $io_cpu); finalize(p)) samples=ns evals=1
    println("r2c_split $n -- CPU      -- $timer_cpu")
    println("r2c_split $n -- PLAN_CPU -- $timer_plan_cpu")

    x_gpu  = ROCVector{Float64}(x_cpu)
    yr_gpu = ROCVector{Float64}(undef, m)
    yi_gpu = ROCVector{Float64}(undef, m)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                          in_type  = rocFFT.rocfft_array_type_real,
                          out_type = rocFFT.rocfft_array_type_hermitian_planar)
    in_ptrs  = [Ptr{Cvoid}(pointer(x_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(yr_gpu)), Ptr{Cvoid}(pointer(yi_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_forward, n;
                       in_type  = rocFFT.rocfft_array_type_real,
                       out_type = rocFFT.rocfft_array_type_hermitian_planar)
        finalize(p)
    end
    println("r2c_split $n -- GPU      -- $timer_gpu")
    println("r2c_split $n -- PLAN_GPU -- $timer_plan_gpu")
    println("r2c_split $n -- RATIO    -- $(timer_cpu / timer_gpu)")

    finalize(plan_cpu); finalize(plan_gpu)
    for buf in (x_gpu, yr_gpu, yi_gpu); AMDGPU.unsafe_free!(buf); end
end

### C2R — split/planar ###
println("\n=== C2R (hermitian_planar / split → real) ===")
for n in sizes
    nw, ns = bench_params(n)
    m = n ÷ 2 + 1
    # build a valid hermitian spectrum
    tmp = rand(Float64, n)
    ro_tmp = Vector{Float64}(undef, m); io_tmp = Vector{Float64}(undef, m)
    p_init = plan_split_dft_r2c(n, tmp, ro_tmp, io_tmp); execute!(p_init, tmp, ro_tmp, io_tmp); finalize(p_init)
    ri_cpu = ro_tmp;  ii_cpu = io_tmp
    out_cpu = Vector{Float64}(undef, n)
    plan_cpu = plan_split_dft_c2r(n, ri_cpu, ii_cpu, out_cpu)
    execute!(plan_cpu, ri_cpu, ii_cpu, out_cpu)
    timer_cpu      = @belapsed execute!($plan_cpu, $ri_cpu, $ii_cpu, $out_cpu) samples=ns evals=1
    timer_plan_cpu = @belapsed (p = plan_split_dft_c2r($n, $ri_cpu, $ii_cpu, $out_cpu); finalize(p)) samples=ns evals=1
    println("c2r_split $n -- CPU      -- $timer_cpu")
    println("c2r_split $n -- PLAN_CPU -- $timer_plan_cpu")

    yr_gpu  = ROCVector{Float64}(ri_cpu)
    yi_gpu  = ROCVector{Float64}(ii_cpu)
    out_gpu = ROCVector{Float64}(undef, n)
    plan_gpu = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                          in_type  = rocFFT.rocfft_array_type_hermitian_planar,
                          out_type = rocFFT.rocfft_array_type_real)
    in_ptrs  = [Ptr{Cvoid}(pointer(yr_gpu)), Ptr{Cvoid}(pointer(yi_gpu))]
    out_ptrs = [Ptr{Cvoid}(pointer(out_gpu))]
    timer_gpu      = gpu_belapsed(() -> execute!(plan_gpu, in_ptrs, out_ptrs); nwarmup=nw, nsamples=ns)
    timer_plan_gpu = gpu_belapsed(nwarmup=nw, nsamples=ns) do
        p = RocFFTPlan(rocFFT.rocfft_transform_type_real_inverse, n;
                       in_type  = rocFFT.rocfft_array_type_hermitian_planar,
                       out_type = rocFFT.rocfft_array_type_real)
        finalize(p)
    end
    println("c2r_split $n -- GPU      -- $timer_gpu")
    println("c2r_split $n -- PLAN_GPU -- $timer_plan_gpu")
    println("c2r_split $n -- RATIO    -- $(timer_cpu / timer_gpu)")

    finalize(plan_cpu); finalize(plan_gpu)
    for buf in (yr_gpu, yi_gpu, out_gpu); AMDGPU.unsafe_free!(buf); end
end
