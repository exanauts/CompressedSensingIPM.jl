using DelimitedFiles, Plots

# ─── Load CSV ────────────────────────────────────────────────────────────────

data, hdr = readdlm("results_rocfft.csv", ',', header=true)
hdr = vec(hdr)

col(name) = findfirst(==(name), hdr)

transform_col = data[:, col("transform")]
layout_col    = data[:, col("layout")]
n_all         = Int.(data[:, col("n")])
exe_cpu_all   = Float64.(data[:, col("execute_cpu")])
plan_cpu_all  = Float64.(data[:, col("plan_cpu")])
exe_gpu_all   = Float64.(data[:, col("execute_gpu")])
plan_gpu_all  = Float64.(data[:, col("plan_gpu")])

# ─── Dataset builder ─────────────────────────────────────────────────────────

function extract(transform, layout)
    mask = (transform_col .== transform) .& (layout_col .== layout)
    idx  = sortperm(n_all[mask])   # ensure sorted by n
    rows = findall(mask)[idx]
    (
        n        = n_all[rows],
        exe_cpu  = exe_cpu_all[rows],
        plan_cpu = plan_cpu_all[rows],
        exe_gpu  = exe_gpu_all[rows],
        plan_gpu = plan_gpu_all[rows],
    )
end

# ─── Datasets ────────────────────────────────────────────────────────────────

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

# ─── Plot generator ──────────────────────────────────────────────────────────

exe_ratio(d)  = d.exe_cpu  ./ d.exe_gpu
plan_ratio(d) = d.plan_cpu ./ d.plan_gpu

function make_plots(datasets, prefix, group_title)
    p1 = plot(title="$group_title — execute time CPU vs GPU",
              xlabel="n", ylabel="Time (s)",
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
              xlabel="n", ylabel="CPU time / GPU time",
              xscale=:log2, legend=:topleft, size=(900, 550))
    for (d, label, col, mk) in datasets
        plot!(p2, d.n, exe_ratio(d); label=label, color=col, marker=mk, markersize=4)
    end
    hline!(p2, [1.0]; linestyle=:dot, color=:black, label="breakeven")
    savefig(p2, "$(prefix)_execute_ratio.png")
    println("Saved $(prefix)_execute_ratio.png")

    p3 = plot(title="$group_title — plan creation time CPU vs GPU",
              xlabel="n", ylabel="Time (s)",
              xscale=:log2, yscale=:log10,
              legend=:topleft, size=(900, 550))
    for (d, label, col, mk) in datasets
        plot!(p3, d.n, d.plan_cpu; label="$label CPU", color=col,
              linestyle=:solid, marker=mk, markersize=4)
        plot!(p3, d.n, d.plan_gpu; label="$label GPU", color=col,
              linestyle=:dash, marker=mk, markersize=4)
    end
    savefig(p3, "$(prefix)_plan_time.png")
    println("Saved $(prefix)_plan_time.png")

    p4 = plot(title="$group_title — plan ratio CPU/GPU",
              xlabel="n", ylabel="CPU plan time / GPU plan time",
              xscale=:log2, yscale=:log10,
              legend=:topleft, size=(900, 550))
    for (d, label, col, mk) in datasets
        plot!(p4, d.n, plan_ratio(d); label=label, color=col, marker=mk, markersize=4)
    end
    hline!(p4, [1.0]; linestyle=:dot, color=:black, label="breakeven")
    savefig(p4, "$(prefix)_plan_ratio.png")
    println("Saved $(prefix)_plan_ratio.png")

    pall = plot(p1, p2, p3, p4; layout=(2, 2), size=(1600, 1000))
    savefig(pall, "$(prefix)_all.png")
    println("Saved $(prefix)_all.png")
end

# ─── Generate plots ──────────────────────────────────────────────────────────

make_plots(c2c_datasets,     "plot_c2c",     "C2C")
make_plots(r2c_c2r_datasets, "plot_r2c_c2r", "R2C/C2R")
