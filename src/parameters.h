
#ifndef ARMON_KOKKOS_PARAMETERS_H
#define ARMON_KOKKOS_PARAMETERS_H

#include <vector>
#include <tuple>
#include <cmath>
#include <array>

#include <Kokkos_Core.hpp>

#include "kernels/common.h"
#include "kernels/limiters.h"
#include "kernels/test_cases.h"


enum class Scheme {
    Godunov, GAD
};

enum class Riemann {
    Acoustic
};

enum class Projection {
    None, Euler, Euler_2nd
};

enum class AxisSplitting {
    Sequential, SequentialSym, Strang, X_only, Y_only
};


struct Params
{
    Test test             = Test::Sod;
    Scheme scheme         = Scheme::GAD;
    Riemann riemann       = Riemann::Acoustic;
    Projection projection = Projection::Euler;
    Limiter limiter       = Limiter::Minmod;
    Axis current_axis     = Axis::X;
    AxisSplitting axis_splitting = AxisSplitting::Sequential;

    TestCase* test_case;

    // Domain parameters
    int nb_ghosts = 3;
    int nx = 10;
    int ny = 10;
    flt_t dx = 0;
    flt_t cfl = 0;
    flt_t Dt = 0;
    int stencil_width = 0;
    bool cst_dt = false;
    std::array<flt_t, 2> domain_size = { 0, 0 };
    std::array<flt_t, 2> domain_origin = { NAN, NAN };  // NAN -> use the default value for the current test

    // Indexing

    // Dimensions of an array
    int row_length = 0;
    int col_length = 0;
    int nb_cells = 0;
    // First and last index of the real domain of an array
    int ideb = 0;
    int ifin = 0;
    int index_start = 0;
    // Used only for indexing with a 2-dimensional index
    int idx_row = 0;
    int idx_col = 0;

    int s = 0; // Stride

    // Computation bounds
    int max_cycles = 100;
    flt_t max_time = 0;

    // Output
    bool write_output = false;
    bool write_ghosts = false;
    bool write_throughput = false;
    int output_precision = 9;
    int verbose = 2;

    const char* output_file = "output_cpp";

    // Comparison
    bool compare = false;

    void init();
    void set_default_values();
    void init_indexing();
    [[nodiscard]] bool check() const;
    void print() const;

    void update_axis(Axis axis);
    [[nodiscard]] std::vector<std::pair<Axis, flt_t>> split_axes(int cycle) const;
};

#endif //ARMON_KOKKOS_PARAMETERS_H
