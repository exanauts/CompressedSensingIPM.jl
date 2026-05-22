using DelimitedFiles, Plots

# Dimension: 1, 2, or 3  (pass as command-line argument, default 1)
#   julia plot_results.jl      → plots results_rocfft_1d.csv
#   julia plot_results.jl 2    → plots results_rocfft_2d.csv
#   julia plot_results.jl 3    → plots results_rocfft_3d.csv
const DIM = isempty(ARGS) ? 1 : parse(Int, ARGS[1])
@assert DIM in (1, 2, 3) "DIM must be 1, 2, or 3"

# ─── Load CSV ────────────────────────────────────────────────────────────────

infile = "results_rocfft_$(DIM)d.csv"
data, hdr = readdlm(infile, ',', header=true)
hdr = vec(hdr)
println("Loaded $infile — $(size(data,1)) rows")

col(name) = findfirst(==(name), hdr)

transform_col = data[:, col("transform")]
layout_col    = data[:, col("layout")]
n_all         = Int.(data[:, col("n")])          # total elements
exe_cpu_all   = Float64.(data[:, col("execute_cpu")])
plan_cpu_all  = Float64.(data[:, col("plan_cpu")])
exe_gpu_all   = Float64.(data[:, col("execute_gpu")])
plan_gpu_all  = Float64.(data[:, col("plan_gpu")])

# For 2D/3D convert total n to side length for a cleaner x-axis
side_all = if DIM == 1
    n_all
elseif DIM == 2
    Int.(round.(sqrt.(n_all)))
else
    Int.(round.(cbrt.(n_all)))
end

xlabel_str = DIM == 1 ? "n" : "side length n  (total = n^$DIM)"

# ─── Dataset builder ─────────────────────────────────────────────────────────

function extract(transform, layout)
    mask = (transform_col .== transform) .& (layout_col .== layout)
    any(mask) || return nothing
    idx  = sortperm(side_all[mask])
    rows = findall(mask)[idx]
    (
        n        = side_all[rows],
        exe_cpu  = exe_cpu_all[rows],
        plan_cpu = plan_cpu_all[rows],
        exe_gpu  = exe_gpu_all[rows],
        plan_gpu = plan_gpu_all[rows],
    )
end

# ─── Datasets ────────────────────────────────────────────────────────────────

if DIM == 1
    c2c_datasets = [
        (extract("c2c_fwd", "interleaved"), "C2C fwd interleaved", :blue,   :circle),
        (extract("c2c_inv", "interleaved"), "C2C inv interleaved", :red,    :square),
        (extract("c2c_fwd", "split"),       "C2C fwd split",       :green,  :diamond),
        (extract("c2c_inv", "split"),       "C2C inv split",       :orange, :utriangle),
    ]

    r2c_c2r_datasets = [
        (extract("r2c", "interleaved"), "R2C interleaved", :blue,   :circle),
        (extract("c2r", "interleaved"), "C2R interleaved", :red,    :square),
        (extract("r2c", "split"),       "R2C split",       :green,  :diamond),
        (extract("c2r", "split"),       "C2R split",       :orange, :utriangle),
    ]
else
    # 2D/3D: interleaved and split
    c2c_datasets = [
        (extract("c2c_fwd", "interleaved"), "C2C fwd interleaved", :blue,   :circle),
        (extract("c2c_inv", "interleaved"), "C2C inv interleaved", :red,    :square),
        (extract("c2c_fwd", "split"),       "C2C fwd split",       :green,  :diamond),
        (extract("c2c_inv", "split"),       "C2C inv split",       :orange, :utriangle),
    ]

    r2c_c2r_datasets = [
        (extract("r2c", "interleaved"), "R2C interleaved", :blue,   :circle),
        (extract("c2r", "interleaved"), "C2R interleaved", :red,    :square),
        (extract("r2c", "split"),       "R2C split",       :green,  :diamond),
        (extract("c2r", "split"),       "C2R split",       :orange, :utriangle),
    ]
end

# Filter out missing datasets
filter!(x -> x[1] !== nothing, c2c_datasets)
filter!(x -> x[1] !== nothing, r2c_c2r_datasets)

# ─── Plot generator ──────────────────────────────────────────────────────────

exe_ratio(d)  = d.exe_cpu  ./ d.exe_gpu
plan_ratio(d) = d.plan_cpu ./ d.plan_gpu

function make_plots(datasets, prefix, group_title)
    p1 = plot(title="$group_title — execute time CPU vs GPU",
              xlabel=xlabel_str, ylabel="Time (s)",
              xscale=:log2, yscale=:log10,
              legend=:topleft, size=(900, 550))
    for (d, label, col, mk) in datasets
        plot!(p1, d.n, d.exe_cpu; label="$label CPU", color=col,
              linestyle=:solid, marker=mk, markersize=4)
        plot!(p1, d.n, d.exe_gpu; label="$label GPU", color=col,
              linestyle=:dash, marker=mk, markersize=4)
    end
    savefig(p1, "$(prefix)_execute_time.png")
    println("Saved $(prefix)_execute_time.png")

    p2 = plot(title="$group_title — execute ratio CPU/GPU  (>1 = GPU faster)",
              xlabel=xlabel_str, ylabel="CPU time / GPU time",
              xscale=:log2, legend=:topleft, size=(900, 550))
    for (d, label, col, mk) in datasets
        plot!(p2, d.n, exe_ratio(d); label=label, color=col, marker=mk, markersize=4)
    end
    hline!(p2, [1.0]; linestyle=:dot, color=:black, label="breakeven")
    savefig(p2, "$(prefix)_execute_ratio.png")
    println("Saved $(prefix)_execute_ratio.png")

    p3_cpu = plot(title="$group_title — plan creation time (CPU)",
                  xlabel=xlabel_str, ylabel="Time (s)",
                  xscale=:log2, yscale=:log10,
                  legend=:topleft, size=(750, 500))
    p3_gpu = plot(title="$group_title — plan creation time (GPU)",
                  xlabel=xlabel_str, ylabel="Time (s)",
                  xscale=:log2, yscale=:log10,
                  legend=:topleft, size=(750, 500))
    for (d, label, col, mk) in datasets
        plot!(p3_cpu, d.n, d.plan_cpu; label=label, color=col, marker=mk, markersize=4)
        plot!(p3_gpu, d.n, d.plan_gpu; label=label, color=col, marker=mk, markersize=4)
    end
    p3 = plot(p3_cpu, p3_gpu; layout=(1, 2), size=(1400, 500))
    savefig(p3, "$(prefix)_plan_time.png")
    println("Saved $(prefix)_plan_time.png")

    p4 = plot(title="$group_title — plan ratio CPU/GPU",
              xlabel=xlabel_str, ylabel="CPU plan time / GPU plan time",
              xscale=:log2, yscale=:log10,
              legend=:topleft, size=(900, 550))
    for (d, label, col, mk) in datasets
        plot!(p4, d.n, plan_ratio(d); label=label, color=col, marker=mk, markersize=4)
    end
    hline!(p4, [1.0]; linestyle=:dot, color=:black, label="breakeven")
    savefig(p4, "$(prefix)_plan_ratio.png")
    println("Saved $(prefix)_plan_ratio.png")

    pall = plot(p1, p2, p3_cpu, p3_gpu, p4, plot(); layout=(3, 2), size=(1600, 1400))
    savefig(pall, "$(prefix)_all.png")
    println("Saved $(prefix)_all.png")
end

# ─── Interleaved / Split ratio tables and plots (1D only) ────────────────────

function layout_comparison(int_data, split_data, label)
    @assert int_data.n == split_data.n "n mismatch for $label"
    ns          = int_data.n
    ratio_cpu   = int_data.exe_cpu ./ split_data.exe_cpu
    ratio_gpu   = int_data.exe_gpu ./ split_data.exe_gpu

    println("\n=== $label : interleaved / split ===")
    println("  (<1 = interleaved faster,  >1 = split faster)")
    println(rpad("n", 12), rpad("CPU ratio", 14), "GPU ratio")
    println("-" ^ 38)
    for i in eachindex(ns)
        println(rpad(string(ns[i]), 12),
                rpad(round(ratio_cpu[i]; digits=3), 14),
                round(ratio_gpu[i]; digits=3))
    end

    (ratio_cpu=ratio_cpu, ratio_gpu=ratio_gpu, n=ns)
end

function plot_layout_ratios(comparisons, prefix)
    p_cpu = plot(title="Interleaved / Split — execute time CPU\n(<1 = interleaved faster)",
                 xlabel=xlabel_str, ylabel="ratio",
                 xscale=:log2, legend=:topleft, size=(900, 550))
    p_gpu = plot(title="Interleaved / Split — execute time GPU\n(<1 = interleaved faster)",
                 xlabel=xlabel_str, ylabel="ratio",
                 xscale=:log2, legend=:topleft, size=(900, 550))

    colors = [:blue, :red, :green, :orange]
    markers = [:circle, :square, :diamond, :utriangle]

    for (i, (label, r)) in enumerate(comparisons)
        plot!(p_cpu, r.n, r.ratio_cpu; label=label, color=colors[i],
              marker=markers[i], markersize=4)
        plot!(p_gpu, r.n, r.ratio_gpu; label=label, color=colors[i],
              marker=markers[i], markersize=4)
    end
    hline!(p_cpu, [1.0]; linestyle=:dot, color=:black, label="parity")
    hline!(p_gpu, [1.0]; linestyle=:dot, color=:black, label="parity")

    p = plot(p_cpu, p_gpu; layout=(1, 2), size=(1600, 550))
    savefig(p, "$(prefix)_layout_ratio.png")
    println("Saved $(prefix)_layout_ratio.png")
end

# ─── Generate plots ──────────────────────────────────────────────────────────

make_plots(c2c_datasets,     "plot_$(DIM)d_c2c",     "$(DIM)D C2C")
make_plots(r2c_c2r_datasets, "plot_$(DIM)d_r2c_c2r", "$(DIM)D R2C/C2R")

# Layout comparison (interleaved / split)
let pfx = "plot_$(DIM)d"
    c2c_fwd_int_d   = extract("c2c_fwd", "interleaved")
    c2c_fwd_split_d = extract("c2c_fwd", "split")
    c2c_inv_int_d   = extract("c2c_inv", "interleaved")
    c2c_inv_split_d = extract("c2c_inv", "split")

    r_c2c_fwd = layout_comparison(c2c_fwd_int_d, c2c_fwd_split_d, "C2C fwd")
    r_c2c_inv = layout_comparison(c2c_inv_int_d, c2c_inv_split_d, "C2C inv")
    plot_layout_ratios([("C2C fwd", r_c2c_fwd), ("C2C inv", r_c2c_inv)], "$(pfx)_c2c")

    r2c_int_d   = extract("r2c", "interleaved")
    r2c_split_d = extract("r2c", "split")
    c2r_int_d   = extract("c2r", "interleaved")
    c2r_split_d = extract("c2r", "split")

    r_r2c = layout_comparison(r2c_int_d, r2c_split_d, "R2C")
    r_c2r = layout_comparison(c2r_int_d, c2r_split_d, "C2R")
    plot_layout_ratios([("R2C", r_r2c), ("C2R", r_c2r)], "$(pfx)_r2c_c2r")
end
