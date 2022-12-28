
#include "parameters.h"

#include <cstdio>

#include "Kokkos_Core.hpp"


void Params::init_indexing() {
    // Dimensions of an array
    row_length = nb_ghosts * 2 + nx;
    col_length = nb_ghosts * 2 + ny;
    nb_cells = row_length * col_length;

    // First and last index of the real domain of an array
    ideb = row_length * nb_ghosts + nb_ghosts;
    ifin = row_length * (ny - 1 + nb_ghosts) + nb_ghosts + nx - 1;
    index_start = ideb;

    // Used only for indexing with a 2 dimensional index
    idx_row = row_length;
    idx_col = 1;

    if (single_comm_per_axis_pass) {
        extra_ring_width = 1;
        extra_ring_width += projection == Projection::Euler_2nd;
    } else {
        extra_ring_width = 0;
    }

    int min_ghosts = 1;
    min_ghosts += scheme == Scheme::GAD;
    min_ghosts += single_comm_per_axis_pass;
    min_ghosts += projection == Projection::Euler_2nd;
    if (nb_ghosts < min_ghosts) {
        std::cerr << "WARNING: There is not enough ghost cells for the given parameters. Expected at least "
            << min_ghosts << ", got " << nb_ghosts << "\n";
    }

    update_axis(current_axis);
}


void Params::print() const
{
    printf("Parameters:\n");
#ifdef KOKKOS_ENABLE_OPENMP
    int max_num_threads = Kokkos::OpenMP::concurrency();
        printf(" - multithreading: 1, (%d threads)\n", max_num_threads);
#else
    printf(" - multithreading: 0\n");
#endif
    printf(" - use simd:   %d\n", USE_SIMD);
#ifdef KOKKOS_ENABLE_CUDA
    printf(" - use gpu:    %d\n", Kokkos::Cuda::impl_is_initialized());
#elif defined(KOKKOS_ENABLE_HIP)
    printf(" - use gpu:    %d\n", Kokkos::Experimental::HIP::impl_is_initialized());
#else
    printf(" - use gpu:    %d\n", 0);
#endif
    printf(" - ieee bits:  %lu\n", 8 * sizeof(flt_t));
    printf("\n");
    printf(" - test:       ");
    switch (test) {
    case Test::Sod:       printf("Sod X\n");           break;
    case Test::Sod_y:     printf("Sod Y\n");           break;
    case Test::Sod_circ:  printf("Cylindrical Sod\n"); break;
    case Test::Bizarrium: printf("Bizarrium\n");       break;
    }
    printf(" - riemann:    %s\n", "acoustic");
    printf(" - scheme:     %s, ", (scheme == Scheme::Godunov) ? "Godunov" : "GAD");
    switch (limiter) {
    case Limiter::None:     printf("no limiter\n");       break;
    case Limiter::Minmod:   printf("Minmod limiter\n");   break;
    case Limiter::Superbee: printf("Superbee limiter\n"); break;
    }
    printf(" - domain:     %dx%d (%d ghosts)\n", nx, ny, nb_ghosts);
    printf(" - nb cells:   %g (%g total)\n", double(nx * ny), double(nb_cells));
    printf(" - CFL:        %g\n", cfl);
    printf(" - Dt init:    %g\n", Dt);
    printf(" - cst dt:     %d\n", cst_dt);
    printf(" - projection: ");
    switch (projection) {
    case Projection::None:      printf("None\n");             break;
    case Projection::Euler:     printf("Euler 1st order\n");  break;
    case Projection::Euler_2nd: printf("Euler 2nd order \n"); break;
    }
    printf(" - splitting:  ");
    switch (axis_splitting) {
    case AxisSplitting::Sequential:    printf("Sequential (X,Y)\n");         break;
    case AxisSplitting::SequentialSym: printf("SequentialSym (X,Y,Y,X)\n");  break;
    case AxisSplitting::Strang:        printf("Strang (½X,Y,½X,½Y,X,½Y)\n"); break;
    case AxisSplitting::X_only:        printf("X only\n");                   break;
    case AxisSplitting::Y_only:        printf("Y only\n");                   break;
    }
    printf(" - single comm:%d\n", single_comm_per_axis_pass);
    printf(" - max time:   %g\n", max_time);
    printf(" - max cycles: %d\n", max_cycles);
    if (write_output) {
        printf(" - output:     '%s'\n", output_file);
    }
    else {
        printf(" - no output\n");
    }
    if (compare) {
        printf(" - comparison: true\n");
    }
}


void Params::update_axis(Axis axis)
{
    current_axis = axis;

    switch (axis) {
    case Axis::X:
        s = 1;
        dx = 1 / flt_t(nx);
        return;

    case Axis::Y:
        s = row_length;
        dx = 1 / flt_t(ny);
        return;
    }
}


std::vector<std::pair<Axis, flt_t>> Params::split_axes(int cycle) const
{
    using Axis = Axis;
    Axis axis_1, axis_2;
    if (cycle % 2 == 0) {
        axis_1 = Axis::Y;
        axis_2 = Axis::X;
    }
    else {
        axis_1 = Axis::X;
        axis_2 = Axis::Y;
    }

    switch (axis_splitting) {
    case AxisSplitting::Sequential:
        return {
            {Axis::X, 1.0},
            {Axis::Y, 1.0}
        };

    case AxisSplitting::SequentialSym:
        return {
            {axis_1, 1.0},
            {axis_2, 1.0}
        };

    case AxisSplitting::Strang:
        return {
            {axis_1, 0.5},
            {axis_2, 1.0},
            {axis_1, 0.5}
        };

    case AxisSplitting::X_only:
        return { {Axis::X, 1.0} };

    case AxisSplitting::Y_only:
        return { {Axis::Y, 1.0} };

    default:
        printf("Wrong axis splitting: %d\n", (int) axis_splitting);
        return {};
    }
}
