# CompressedSensingIPM.jl

[![License](https://img.shields.io/github/license/exanauts/CompressedSensingIPM.jl)](https://github.com/exanauts/CompressedSensingIPM.jl/blob/main/LICENSE)
[![CI](https://github.com/exanauts/CompressedSensingIPM.jl/actions/workflows/action.yml/badge.svg)](https://github.com/exanauts/CompressedSensingIPM.jl/actions)

**CompressedSensingIPM.jl** is a Julia package for solving large-scale compressed sensing problems using a matrix-free primal-dual interior point method.
This solver is built on top of [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl) and [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

## Reference

```bibtex
@article{wei-montoison-rao-pacaud-anitescu-2025,
  author  = {Kuang, Wei and Montoison, Alexis and Rao, Vishwas and Pacaud, Fran{\c{c}}ois and Anitescu, Mihai},
  title   = {{Recovering sparse DFT from missing signals via interior point method on GPU}},
  journal = {arXiv preprint arXiv:2502.04217},
  year    = {2025},
  doi     = {10.48550/arXiv.2502.04217}
}
```

## Installation

CompressedSensingIPM.jl can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add https://github.com/exanauts/CompressedSensingIPM.jl.git
pkg> test CompressedSensingIPM
```

## Funding

This research was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division.
This research used resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (DOE) Office of Science user facility at Argonne National Laboratory, and is based on research supported by the U.S. DOE Office of Science-Advanced Scientific Computing Research Program, under Contract No DE-AC02-06CH11357.
