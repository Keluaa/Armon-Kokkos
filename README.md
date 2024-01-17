# Armon-Kokkos

C++ mirror of the [Julia solver](https://github.com/Keluaa/Armon.jl), written using the [Kokkos library](https://github.com/kokkos/kokkos).

[Kernels](src/kernels) are made to be called from both the C++ and Julia solvers, allowing finer performance comparisons.

In Julia, the Kokkos environment is managed by [Kokkos.jl](https://github.com/Keluaa/Kokkos.jl).

The [`benchmark`](https://github.com/Keluaa/Armon-Kokkos/tree/benchmark) branch is the state of the repository for which most of the results for an upcoming paper were obtained.

