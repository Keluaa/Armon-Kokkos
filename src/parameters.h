
#ifndef ARMON_KOKKOS_PARAMETERS_H
#define ARMON_KOKKOS_PARAMETERS_H

#include <vector>
#include <tuple>
#include <cmath>


#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#define USE_SIMD 1
#else
#define USE_SIMD 0
#endif


#if USE_SINGLE_PRECISION
typedef float flt_t;
#else
typedef double flt_t;
#endif

enum class Side {
    Left,
    Right,
    Top,
    Bottom
};

enum class Test {
    Sod, Sod_y, Sod_circ, Bizarrium
};

std::array<flt_t, 2> default_domain_size(Test test);
std::array<flt_t, 2> default_domain_origin(Test test);
flt_t default_CFL(Test test);
flt_t default_max_time(Test test);

struct TestInitParams {
    flt_t gamma, high_rho, low_rho, high_p, low_p, high_u, low_u, high_v, low_v;
};

TestInitParams get_test_init_params(Test test);

bool test_region_high(Test test, flt_t x, flt_t y);
std::array<flt_t, 2> boundaryCondition(Test test, Side side);


enum class Scheme {
    Godunov, GAD
};

enum class Riemann {
    Acoustic
};

enum class Projection {
    None, Euler, Euler_2nd
};

enum class Limiter {
    None, Minmod, Superbee
};

enum class AxisSplitting {
    Sequential, SequentialSym, Strang, X_only, Y_only
};

enum class Axis {
    X, Y
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

    // Domain parameters
    int nb_ghosts = 3;
    int nx = 10;
    int ny = 10;
    flt_t dx = 0;
    flt_t cfl = 0;
    flt_t Dt = 0;
    int stencil_width = 0;
    bool cst_dt = false;
    bool single_comm_per_axis_pass = false;
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
    // Used only for indexing with a 2 dimensional index
    int idx_row = 0;
    int idx_col = 0;
    int extra_ring_width = 0;

    int s = 0; // Stride

    // Computation bounds
    int max_cycles = 100;
    flt_t max_time = 0;

    // Output
    bool write_output = false;
    bool write_ghosts = false;
    bool write_throughput = false;
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
