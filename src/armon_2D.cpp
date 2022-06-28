
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <string>
#include <array>

#include <Kokkos_Core.hpp>


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


using view = Kokkos::View<flt_t*>;
using host_view = view::HostMirror;

using dim3d = std::array<unsigned int, 3>;


struct Params
{
    // Test and solver
    enum class Test {
        Sod, Sod_y, Sod_circ, Bizarrium
    } test = Test::Sod;

    enum class Scheme {
        Godunov, GAD_minmod
    } scheme = Scheme::GAD_minmod;

    enum class Riemann {
        Acoustic
    } riemann = Riemann::Acoustic;

    // Domain parameters
    int nb_ghosts = 2;
    int nx = 10;
    int ny = 10;
    flt_t dx = 0;
    flt_t cfl = 0.6;
    flt_t Dt = 0.0;
    bool cst_dt = false;
    bool euler_projection = false;
    bool transpose_dims = false;

    enum class AxisSplitting {
        Sequential, SequentialSym, Strang
    } axis_splitting = AxisSplitting::Sequential;

    // Indexing

    // Dimensions of an array
    int row_length = 0;
    int col_length = 0;
    int nb_cells = 0;
    // First and last index of the real domain of an array
    int ideb = 0;
    int ifin = 0;
    int index_start = 0;
    // Same as above, but for a transposed array
    int ideb_T = 0;
    int ifin_T = 0;
    int index_start_T = 0;
    // Used only for indexing with a 2 dimensional index
    int idx_row = 0;
    int idx_col = 0;

    enum class Axis {
        X, Y
    } current_axis = Axis::X;

    int s = 0; // Stride

    // Compute bounds
    int max_cycles = 100;
    flt_t max_time = 0.0;

    // Output
    bool write_output = false;
    bool write_ghosts = false;
    bool write_throughput = false;
    int verbose = 2;

    const char* output_file = "output_cpp";

    void init_indexing()
    {
        // Dimensions of an array
        row_length = nb_ghosts * 2 + nx;
        col_length = nb_ghosts * 2 + ny;
        nb_cells = row_length * col_length;
        // First and last index of the real domain of an array
        ideb = row_length * nb_ghosts + nb_ghosts;
        ifin = row_length * (ny - 1 + nb_ghosts) + nb_ghosts + nx - 1;
        index_start = ideb;
        // Same as above, but for a transposed array
        ideb_T = col_length * nb_ghosts + nb_ghosts;
        ifin_T = col_length * (nx - 1 + nb_ghosts) + nb_ghosts + ny - 1;
        index_start_T = ideb_T;
        // Used only for indexing with a 2 dimensional index
        idx_row = row_length;
        idx_col = 1;
    }

    void print() const
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
        printf(" - use gpu:    %d\n", Kokkos::HIP::impl_is_initialized());
#else
        printf(" - use gpu:    %d\n", 0);
#endif
        printf(" - ieee bits:  %lu\n", 8 * sizeof(flt_t));
        printf("\n");
        printf(" - test:       ");
        switch (test) {
        case Test::Sod:       printf("Sod X\n"); break;
        case Test::Sod_y:     printf("Sod Y\n"); break;
        case Test::Sod_circ:  printf("Cylindrical Sod\n"); break;
        case Test::Bizarrium: printf("Bizarrium\n"); break;
        }
        printf(" - riemann:    %s\n", "acoustic");
        printf(" - scheme:     %s\n", (scheme == Scheme::Godunov) ? "Godunov" : "GAD-minmod");
        printf(" - domain:     %dx%d (%d ghosts)\n", nx, ny, nb_ghosts);
        printf(" - nb cells:   %g (%g total)\n", double(nx * ny), double(nb_cells));
        printf(" - CFL:        %g\n", cfl);
        printf(" - Dt init:    %g\n", Dt);
        printf(" - cst dt:     %d\n", cst_dt);
        printf(" - euler proj: %d\n", euler_projection);
        printf(" - trans dims: %d\n", transpose_dims);
        printf(" - splitting:  ");
        switch (axis_splitting) {
        case AxisSplitting::Sequential:    printf("Sequential\n"); break;
        case AxisSplitting::SequentialSym: printf("SequentialSym\n"); break;
        case AxisSplitting::Strang:        printf("Strang\n"); break;
        }
        printf(" - max time:   %g\n", max_time);
        printf(" - max cycles: %d\n", max_cycles);
        if (write_output) {
            printf(" - output:     '%s'\n", output_file);
        }
        else {
            printf(" - no output\n");
        }
    }
};


template<typename view_t>
struct DataHolder
{
    view_t x, y;
    view_t rho, umat, vmat, Emat, pmat, cmat, gmat, ustar, pstar, ustar_1, pstar_1;
    view_t tmp_rho, tmp_urho, tmp_vrho, tmp_Erho;
    view_t domain_mask, domain_mask_T;

    DataHolder() = default;

    [[maybe_unused]]
    DataHolder(const std::string& label, int size)
            : x(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , y(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , rho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , umat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , vmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , Emat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , pmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , cmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , gmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , ustar(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , pstar(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , ustar_1(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , pstar_1(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , tmp_rho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , tmp_urho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , tmp_vrho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , tmp_Erho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , domain_mask(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
            , domain_mask_T(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
    { }

    [[nodiscard]]
    DataHolder<typename view_t::HostMirror> as_mirror() const
    {
        DataHolder<typename view_t::HostMirror> mirror;
        mirror.x = Kokkos::create_mirror_view(x);
        mirror.y = Kokkos::create_mirror_view(y);
        mirror.rho = Kokkos::create_mirror_view(rho);
        mirror.umat = Kokkos::create_mirror_view(umat);
        mirror.vmat = Kokkos::create_mirror_view(vmat);
        mirror.Emat = Kokkos::create_mirror_view(Emat);
        mirror.pmat = Kokkos::create_mirror_view(pmat);
        mirror.cmat = Kokkos::create_mirror_view(cmat);
        mirror.gmat = Kokkos::create_mirror_view(gmat);
        mirror.ustar = Kokkos::create_mirror_view(ustar);
        mirror.pstar = Kokkos::create_mirror_view(pstar);
        mirror.ustar_1 = Kokkos::create_mirror_view(ustar_1);
        mirror.pstar_1 = Kokkos::create_mirror_view(pstar_1);
        mirror.tmp_rho = Kokkos::create_mirror_view(tmp_rho);
        mirror.tmp_urho = Kokkos::create_mirror_view(tmp_urho);
        mirror.tmp_vrho = Kokkos::create_mirror_view(tmp_vrho);
        mirror.tmp_Erho = Kokkos::create_mirror_view(tmp_Erho);
        mirror.domain_mask = Kokkos::create_mirror_view(domain_mask);
        mirror.domain_mask_T = Kokkos::create_mirror_view(domain_mask_T);
        return mirror;
    }

    template<typename mirror_view_t>
    void deep_copy_to_mirror(DataHolder<mirror_view_t>& mirror) const
    {
        Kokkos::deep_copy(mirror.x, x);
        Kokkos::deep_copy(mirror.y, y);
        Kokkos::deep_copy(mirror.rho, rho);
        Kokkos::deep_copy(mirror.umat, umat);
        Kokkos::deep_copy(mirror.vmat, vmat);
        Kokkos::deep_copy(mirror.Emat, Emat);
        Kokkos::deep_copy(mirror.pmat, pmat);
        Kokkos::deep_copy(mirror.cmat, cmat);
        Kokkos::deep_copy(mirror.gmat, gmat);
        Kokkos::deep_copy(mirror.ustar, ustar);
        Kokkos::deep_copy(mirror.pstar, pstar);
        Kokkos::deep_copy(mirror.ustar_1, ustar_1);
        Kokkos::deep_copy(mirror.pstar_1, pstar_1);
        Kokkos::deep_copy(mirror.tmp_rho, tmp_rho);
        Kokkos::deep_copy(mirror.tmp_urho, tmp_urho);
        Kokkos::deep_copy(mirror.tmp_vrho, tmp_vrho);
        Kokkos::deep_copy(mirror.tmp_Erho, tmp_Erho);
        Kokkos::deep_copy(mirror.domain_mask, domain_mask);
        Kokkos::deep_copy(mirror.domain_mask_T, domain_mask_T);
    }
};


using Data = DataHolder<view>;
using HostData = DataHolder<host_view>;


// Program time contribution tracking
std::map<std::string, double> time_contribution;
#define CAT(a, b) a##b
#define TIC_IMPL(line_nb) auto CAT(tic_, line_nb) = std::chrono::steady_clock::now()
#define TAC_IMPL(label, line_nb) \
    auto CAT(tac_, line_nb) = std::chrono::steady_clock::now(); \
    double CAT(expr_time_, line_nb) = std::chrono::duration<double>(CAT(tac_, line_nb) - CAT(tic_, line_nb)).count(); \
    time_contribution[label]   += CAT(expr_time_, line_nb); \
    time_contribution["TOTAL"] += CAT(expr_time_, line_nb)
#define TIC() TIC_IMPL(__LINE__)
#define TAC(label) TAC_IMPL(label, __LINE__)


KOKKOS_INLINE_FUNCTION int index_T(const Params& p, const int i)
{
    // 'i' is at max N*M, then N*N*M, or N³ for square domains, must be less than 2^31 in order to prevent integer
    // overflow, so N < ∛(2^31)=1290, which is quite limiting.
    // Therefore, we are forced to cast 'i' to a 64-bit integer for this calculation.
    return static_cast<int>((p.col_length * static_cast<long long>(i)) % (p.row_length * p.col_length - 1));
}


KOKKOS_INLINE_FUNCTION int index_1D(const Params& p, const int i, const int j)
{
    return p.index_start + j * p.idx_row + i * p.idx_col;
}


void perfectGasEOS(const Params& p, Data& d, flt_t gamma)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin+1),
    KOKKOS_LAMBDA(const int i) {
        flt_t e = d.Emat[i] - 0.5 * (std::pow(d.umat[i], flt_t(2)) + std::pow(d.vmat[i], flt_t(2)));
        d.pmat[i] = (gamma - 1) * d.rho[i] * e;
        d.cmat[i] = std::sqrt(gamma * d.pmat[i] / d.rho[i]);
        d.gmat[i] = (1 + gamma) / 2;
    });
}


void bizarriumEOS(const Params& p, Data& d)
{
    const flt_t rho0 = 1e4, K0 = 1e11, Cv0 = 1e3, T0 = 300, eps0 = 0;
    const flt_t G0 = 1.5, s = 1.5, q = -42080895./14941154., r = 727668333./149411540.;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin+1),
    KOKKOS_LAMBDA(const int i){
        flt_t x = d.rho[i] / rho0 - 1;
        flt_t g = G0 * (1 - rho0 / d.rho[i]);

        flt_t f0 = (1+(s/3-2)*x+q*(x*x)+r*(x*x*x))/(1-s*x);
        flt_t f1 = (s/3-2+2*q*x+3*r*(x*x)+s*f0)/(1-s*x);
        flt_t f2 = (2*q+6*r*x+2*s*f1)/(1-s*x);
        flt_t f3 = (6*r+3*s*f2)/(1-s*x);

        flt_t eps_k0 = eps0 - Cv0*T0*(1+g) + flt_t(0.5)*(K0/rho0)*(x*x)*f0;
        flt_t pk0 = -Cv0*T0*G0*rho0 + (flt_t(0.5)*K0*x*(1+x)*(1+x)*(2*f0+x*f1));
        flt_t pk0prime = -flt_t(0.5) * K0 * std::pow(1+x,flt_t(3))
                * rho0 * (2 * (1+3*x) * f0 + 2*x*(2+3*x) * f1 + (x*x) * (1+x) * f2);
        flt_t pk0second = flt_t(0.5) * K0 * std::pow(1+x,flt_t(4)) * (rho0*rho0)
                * (12*(1+2*x)*f0 + 6*(1+6*x+6*(x*x)) * f1 + 6*x*(1+x)*(1+2*x) * f2
                   + std::pow(x*(1+x),flt_t(2)) * f3);

        flt_t e = d.Emat[i] - 0.5 * (std::pow(d.umat[i], flt_t(2)) + std::pow(d.vmat[i], flt_t(2)));
        d.pmat[i] = pk0 + G0*rho0*(e - eps_k0);
        d.cmat[i] = std::sqrt(G0*rho0*(d.pmat[i] - pk0) - pk0prime) / d.rho[i];
        d.gmat[i] = flt_t(0.5) / (std::pow(d.rho[i],flt_t(3)) * std::pow(d.cmat[i],flt_t(2)))
                * (pk0second + std::pow(G0 * rho0,flt_t(2)) * (d.pmat[i]-pk0));
    });
}


std::function<bool(flt_t, flt_t)> get_test_condition_lambda(const Params& p)
{
    switch (p.test) {
    case Params::Test::Sod:       return KOKKOS_LAMBDA(flt_t x, flt_t y) { return x <= flt_t(0.5); };
    case Params::Test::Sod_y:     return KOKKOS_LAMBDA(flt_t x, flt_t y) { return y <= flt_t(0.5); };
    case Params::Test::Sod_circ:  return KOKKOS_LAMBDA(flt_t x, flt_t y) { return std::pow(x - flt_t(0.5), flt_t(2)) + std::pow(y - flt_t(0.5), flt_t(2)) <= flt_t(0.125); };
    case Params::Test::Bizarrium: return KOKKOS_LAMBDA(flt_t x, flt_t y) { return x <= 0.5; };
    default:
        fputs("Wrong test", stderr);
        Kokkos::finalize();
        exit(1);
    }
}


void init_test(Params& p, Data& d)
{
    flt_t left_rho, right_rho, left_p, right_p;
    const flt_t gamma = 1.4;

    switch (p.test) {
    case Params::Test::Sod:
    case Params::Test::Sod_y:
    case Params::Test::Sod_circ:
        if (p.max_time == 0.0) p.max_time = 0.20;
        if (p.cfl == 0.0)      p.cfl = 0.95;
        left_rho  = 1.;
        right_rho = 0.125;
        left_p  = 1.0;
        right_p = 0.1;
        break;
    case Params::Test::Bizarrium:
        if (p.max_time == 0.0) p.max_time = 80e-6;
        if (p.cfl == 0.0)      p.cfl = 0.6;
        left_rho  = 1.42857142857e+4;
        right_rho = 10000.;
        left_p  = 1.0;
        right_p = 0.1;
        break;
    }

    auto cond = get_test_condition_lambda(p);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, p.nb_cells),
    KOKKOS_LAMBDA(const int i) {
        int ix = (i % p.row_length) - p.nb_ghosts;
        int iy = (i / p.row_length) - p.nb_ghosts;

        int iT = p.transpose_dims ? index_T(p, i) : i;

        d.x[i]  = flt_t(ix) / flt_t(p.nx);
        d.y[iT] = flt_t(iy) / flt_t(p.ny);

        flt_t x = d.x[i]  + flt_t(1. / (2 * p.nx));
        flt_t y = d.y[iT] + flt_t(1. / (2 * p.ny));
        if (p.test == Params::Test::Bizarrium) {
            if (d.x[i] < 0.5) {
                d.rho[i]  = left_rho;
                d.umat[i] = 0.;
                d.vmat[i] = 0.;
                d.Emat[i] = 4.48657821135e+6;
            }
            else {
                d.rho[i]  = right_rho;
                d.umat[i] = 250.;
                d.vmat[i] = 0.;
                d.Emat[i] = flt_t(0.5) * std::pow(d.umat[i], flt_t(2));
            }
        }
        else {
            if (cond(x, y)) {
                d.rho[i]  = left_rho;
                d.umat[i] = 0.;
                d.vmat[i] = 0.;
                d.Emat[i] = left_p / ((gamma - flt_t(1.)) * d.rho[i]);
            }
            else {
                d.rho[i]  = right_rho;
                d.umat[i] = 0.;
                d.vmat[i] = 0.;
                d.Emat[i] = right_p / ((gamma - flt_t(1.)) * d.rho[i]);
            }
        }

        d.domain_mask[i] = d.domain_mask_T[iT] = (
            0 <= ix && ix < p.nx &&
            0 <= iy && iy < p.ny
        ) ? 1 : 0;

        d.pmat[i] = 0.;
        d.cmat[i] = 1.;
        d.ustar[i] = 0.;
        d.pstar[i] = 0.;
        d.ustar_1[i] = 0.;
        d.pstar_1[i] = 0.;
    });

    if (p.transpose_dims) {
        d.y[0] = d.y[p.col_length];
        d.y[p.nb_cells - 1] = d.y[p.nb_cells - p.col_length - 1];
        d.domain_mask_T[0] = d.domain_mask[0];
        d.domain_mask_T[p.nb_cells - 1] = d.domain_mask[p.nb_cells - 1];
    }

    switch (p.test) {
    case Params::Test::Sod:
    case Params::Test::Sod_y:
    case Params::Test::Sod_circ:
        perfectGasEOS(p, d, gamma);
        break;
    case Params::Test::Bizarrium:
        bizarriumEOS(p, d);
        break;
    }
}


void boundaryConditions(const Params& p, Data& d)
{
    flt_t u_factor_left = 1., u_factor_right = 1., v_factor_bottom = 1., v_factor_top = 1.;

    switch (p.test) {
    case Params::Test::Sod:
        u_factor_left = -1.;
        u_factor_right = -1.;
        break;
    case Params::Test::Sod_y:
        v_factor_top = -1.;
        v_factor_bottom = -1.;
        break;
    case Params::Test::Sod_circ:
        u_factor_left = -1.;
        u_factor_right = -1.;
        v_factor_top = -1.;
        v_factor_bottom = -1.;
        break;
    case Params::Test::Bizarrium:
        break;
    }

    Kokkos::parallel_for(std::max(p.nx, p.ny),
    KOKKOS_LAMBDA(const int tid) {
        if (tid < p.ny) {
            // Left border
            int idx = index_1D(p, 0, tid);
            int idxm1 = index_1D(p, -1, tid);
            d.rho[idxm1]  = d.rho[idx];
            d.umat[idxm1] = d.umat[idx] * u_factor_left;
            d.vmat[idxm1] = d.vmat[idx];
            d.pmat[idxm1] = d.pmat[idx];
            d.cmat[idxm1] = d.cmat[idx];
            d.gmat[idxm1] = d.gmat[idx];

            // Right border
            idx = index_1D(p, p.nx - 1, tid);
            int idxp1 = index_1D(p, p.nx, tid);
            d.rho[idxp1]  = d.rho[idx];
            d.umat[idxp1] = d.umat[idx] * u_factor_right;
            d.vmat[idxp1] = d.vmat[idx];
            d.pmat[idxp1] = d.pmat[idx];
            d.cmat[idxp1] = d.cmat[idx];
            d.gmat[idxp1] = d.gmat[idx];
        }

        if (tid < p.nx) {
            // Bottom border
            int idx = index_1D(p, tid, 0);
            int idxm1 = index_1D(p, tid, -1);
            d.rho[idxm1]  = d.rho[idx];
            d.umat[idxm1] = d.umat[idx];
            d.vmat[idxm1] = d.vmat[idx] * v_factor_bottom;
            d.pmat[idxm1] = d.pmat[idx];
            d.cmat[idxm1] = d.cmat[idx];
            d.gmat[idxm1] = d.gmat[idx];

            // Top border
            idx = index_1D(p, tid, p.ny - 1);
            int idxp1 = index_1D(p, tid, p.ny);
            d.rho[idxp1]  = d.rho[idx];
            d.umat[idxp1] = d.umat[idx];
            d.vmat[idxp1] = d.vmat[idx] * v_factor_top;
            d.pmat[idxp1] = d.pmat[idx];
            d.cmat[idxp1] = d.cmat[idx];
            d.gmat[idxp1] = d.gmat[idx];
        }
    });
}


void first_order_euler_remap(const Params& p, Data& d, flt_t dt)
{
    int range_start, range_end;
    if (p.transpose_dims) {
        range_start = 1;
        range_end = p.nb_cells - 1;
    }
    else {
        range_start = p.ideb;
        range_end = p.ifin + 1;
    }

    const int s = p.s;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(range_start, range_end),
    KOKKOS_LAMBDA(const int i) {
        flt_t dX = p.dx + dt * (d.ustar[i+s] - d.ustar[i]);
        flt_t L1 =  std::max(flt_t(0.), d.ustar[i])   * dt * d.domain_mask[i];
        flt_t L3 = -std::min(flt_t(0.), d.ustar[i+s]) * dt * d.domain_mask[i];
        flt_t L2 = dX - L1 - L3;

        d.tmp_rho[i]  = (L1*d.rho[i-s]             + L2*d.rho[i]           + L3*d.rho[i+s]            ) / dX;
        d.tmp_urho[i] = (L1*d.rho[i-s]*d.umat[i-s] + L2*d.rho[i]*d.umat[i] + L3*d.rho[i+s]*d.umat[i+s]) / dX;
        d.tmp_vrho[i] = (L1*d.rho[i-s]*d.vmat[i-s] + L2*d.rho[i]*d.vmat[i] + L3*d.rho[i+s]*d.vmat[i+s]) / dX;
        d.tmp_Erho[i] = (L1*d.rho[i-s]*d.Emat[i-s] + L2*d.rho[i]*d.Emat[i] + L3*d.rho[i+s]*d.Emat[i+s]) / dX;
    });

    if (p.transpose_dims) {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(range_start, range_end),
        KOKKOS_LAMBDA(const int i) {
            int iT = index_T(p, i);
            d.rho[iT]  = d.tmp_rho[i];
            d.umat[iT] = d.tmp_urho[i] / d.tmp_rho[i];
            d.vmat[iT] = d.tmp_vrho[i] / d.tmp_rho[i];
            d.Emat[iT] = d.tmp_Erho[i] / d.tmp_rho[i];
        });
    }
    else {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(range_start, range_end),
        KOKKOS_LAMBDA(const int i) {
            d.rho[i]  = d.tmp_rho[i];
            d.umat[i] = d.tmp_urho[i] / d.tmp_rho[i];
            d.vmat[i] = d.tmp_vrho[i] / d.tmp_rho[i];
            d.Emat[i] = d.tmp_Erho[i] / d.tmp_rho[i];
        });
    }
}


flt_t dtCFL(const Params& p, Data& d, flt_t dta)
{
    flt_t dt = INFINITY;
    flt_t dx = 1. / p.nx;
    flt_t dy = 1. / p.ny;

    if (p.cst_dt) {
        return p.Dt;
    }
    else if (p.euler_projection) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
        KOKKOS_LAMBDA(const int i, flt_t& dt_loop) {
            flt_t max_cx = std::max(std::abs(d.umat[i] + d.cmat[i]), std::abs(d.umat[i] - d.cmat[i])) * d.domain_mask[i];
            flt_t max_cy = std::max(std::abs(d.vmat[i] + d.cmat[i]), std::abs(d.vmat[i] - d.cmat[i])) * d.domain_mask[i];
            dt_loop = std::min(dt_loop, std::min(dx / max_cx, dy / max_cy));
        }, Kokkos::Min<flt_t>(dt));
    }
    else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
        KOKKOS_LAMBDA(const int i, flt_t& dt_loop) {
            dt_loop = std::min(dt_loop, flt_t(1.) / (d.cmat[i] * d.domain_mask[i]));
        }, Kokkos::Min<flt_t>(dt));
        dt *= std::min(dx, dy);
    }

    if (!std::isfinite(dt) || dt <= 0)
        return dt;
    else if (dta == 0)
        if (p.Dt != 0)
            return p.Dt;
        else
            return p.cfl * dt;
    else
        return std::min(p.cfl * dt, flt_t(1.05) * dta);
}


void acoustic(const Params& p, Data& d, int last_i, const view& u)
{
    const int s = p.s;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, last_i),
    KOKKOS_LAMBDA(const int i){
        flt_t rc_l = d.rho[i-s] * d.cmat[i-s];
        flt_t rc_r = d.rho[i]   * d.cmat[i];
        d.ustar[i] = (rc_l *      u[i-s] + rc_r *      u[i] +               (d.pmat[i-s] - d.pmat[i])) / (rc_l + rc_r);
        d.pstar[i] = (rc_r * d.pmat[i-s] + rc_l * d.pmat[i] + rc_l * rc_r * (     u[i-s] -      u[i])) / (rc_l + rc_r);
    });
}


KOKKOS_INLINE_FUNCTION flt_t phi(flt_t r)
{
    // Minmod
    return std::max(flt_t(0.), std::min(flt_t(1.), r));
}


void acoustic_GAD(const Params& p, Data& d, flt_t dt, int last_i, const view& u)
{
    const int s = p.s;
    const flt_t dx = p.dx;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, last_i),
    KOKKOS_LAMBDA(const int i) {
        flt_t rc_l = d.rho[i-s] * d.cmat[i-s];
        flt_t rc_r = d.rho[i]   * d.cmat[i];
        d.ustar_1[i] = (rc_l *      u[i-s] + rc_r *      u[i] +               (d.pmat[i-s] - d.pmat[i])) / (rc_l + rc_r);
        d.pstar_1[i] = (rc_r * d.pmat[i-s] + rc_l * d.pmat[i] + rc_l * rc_r * (     u[i-s] -      u[i])) / (rc_l + rc_r);
    });

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, last_i),
    KOKKOS_LAMBDA(const int i){
        flt_t r_u_m = (d.ustar_1[i+s] -        u[i]) / (d.ustar_1[i] -      u[i-s] + flt_t(1e-6));
        flt_t r_p_m = (d.pstar_1[i+s] -   d.pmat[i]) / (d.pstar_1[i] - d.pmat[i-s] + flt_t(1e-6));
        flt_t r_u_p = (     u[i-s] - d.ustar_1[i-s]) / (     u[i] -   d.ustar_1[i] + flt_t(1e-6));
        flt_t r_p_p = (d.pmat[i-s] - d.pstar_1[i-s]) / (d.pmat[i] -   d.pstar_1[i] + flt_t(1e-6));

        flt_t dm_l = d.rho[i-s] * dx;
        flt_t dm_r = d.rho[i]   * dx;
        flt_t rc_l = d.rho[i-s] * d.cmat[i-s];
        flt_t rc_r = d.rho[i]   * d.cmat[i];

        flt_t Dm    = (dm_l + dm_r) / 2;
        flt_t theta = (rc_l + rc_r) / 2 * (dt / Dm);

        d.ustar[i] = d.ustar_1[i] + flt_t(0.5) * (1 - theta)
                * (phi(r_u_p) * (     u[i] - d.ustar_1[i]) - phi(r_u_m) * (d.ustar_1[i] -      u[i-s]));
        d.pstar[i] = d.pstar_1[i] + flt_t(0.5) * (1 - theta)
                * (phi(r_p_p) * (d.pmat[i] - d.pstar_1[i]) - phi(r_p_m) * (d.pstar_1[i] - d.pmat[i-s]));
    });
}


void numericalFluxes(const Params& p, Data& d, flt_t dt, int last_i, const view& u)
{
    switch (p.riemann) {
    case Params::Riemann::Acoustic:
    {
        switch (p.scheme) {
        case Params::Scheme::Godunov:    acoustic(p, d, last_i, u); break;
        case Params::Scheme::GAD_minmod: acoustic_GAD(p, d, dt, last_i, u); break;
        }
        break;
    }
    }
}


void cellUpdate(const Params& p, Data& d, flt_t dt, view& u, view& x, const view& domain_mask)
{
    const int s = p.s;
    const flt_t dx = p.dx;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
    KOKKOS_LAMBDA(const int i){
        flt_t mask = domain_mask[i];
        flt_t dm = d.rho[i] * dx;
        d.rho[i]   = dm / (dx + dt * (d.ustar[i+s] - d.ustar[i]) * mask);
        u[i]      += dt / dm * (d.pstar[i]              - d.pstar[i+s]               ) * mask;
        d.Emat[i] += dt / dm * (d.pstar[i] * d.ustar[i] - d.pstar[i+s] * d.ustar[i+s]) * mask;
    });

    if (!p.euler_projection) {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
        KOKKOS_LAMBDA(const int i){
            x[i] += dt * d.ustar[i];
        });
    }
}


void update_EOS(const Params& p, Data& d)
{
    switch (p.test) {
    case Params::Test::Sod:
    case Params::Test::Sod_y:
    case Params::Test::Sod_circ:
    {
        const flt_t gamma = 1.4;
        perfectGasEOS(p, d, gamma);
        break;
    }
    case Params::Test::Bizarrium:
        bizarriumEOS(p, d);
        break;
    }
}


std::tuple<int, view&, view&, view&> update_axis_parameters(Params& p, Data& d, Params::Axis axis)
{
    p.current_axis = axis;

    int last_i = p.ifin + 2;

    switch (axis) {
    case Params::Axis::X:
        p.s = 1;
        p.idx_row = p.row_length;
        p.idx_col = 1;

        p.dx = 1. / p.nx;

        return std::tie(last_i, d.x, d.umat, d.domain_mask);

    case Params::Axis::Y:
        if (p.transpose_dims) {
            p.s = 1;
            p.idx_row = 1;
            p.idx_col = p.row_length;
        }
        else {
            p.s = p.row_length;
            p.idx_row = p.row_length;
            p.idx_col = 1;
        }

        p.dx = 1. / p.ny;
        last_i += p.row_length;

        return std::tie(last_i, d.y, d.vmat, d.domain_mask_T);

    default:
        fputs("Wrong axis", stderr);
        Kokkos::finalize();
        exit(1);
    }
}


void transpose_parameters(Params& p)
{
    std::swap(p.row_length, p.col_length);

    std::swap(p.ideb, p.ideb_T);
    std::swap(p.ifin, p.ifin_T);
    std::swap(p.index_start, p.index_start_T);
}


std::vector<std::pair<Params::Axis, flt_t>> split_axes(const Params& p, int cycle)
{
    using Axis = Params::Axis;
    Axis axis_1, axis_2;
    if (cycle % 2 == 0) {
        axis_1 = Axis::Y;
        axis_2 = Axis::X;
    }
    else {
        axis_1 = Axis::X;
        axis_2 = Axis::Y;
    }

    switch (p.axis_splitting) {
    case Params::AxisSplitting::Sequential:
        return {
            {Axis::X, 1.0},
            {Axis::Y, 1.0}
        };

    case Params::AxisSplitting::SequentialSym:
        return {
            {axis_1, 1.0},
            {axis_2, 1.0}
        };

    case Params::AxisSplitting::Strang:
        return {
            {axis_1, 0.5},
            {axis_2, 1.0},
            {axis_1, 0.5}
        };

    default:
        fputs("Wrong axis splitting", stderr);
        Kokkos::finalize();
        exit(1);
    }
}


void write_output(const Params& p, const HostData& d)
{
    FILE* file = fopen(p.output_file, "w");

    if (p.write_ghosts) {
        for (int j = -p.nb_ghosts; j < p.ny + p.nb_ghosts; j++) {
            for (int i = -p.nb_ghosts; i < p.nx + p.nb_ghosts; i++) {
                int idx_x = index_1D(p, i, j);
                int idx_y = p.transpose_dims ? index_1D(p, j, i) : idx_x;
                fprintf(file, "%f, %f, %f\n", d.x[idx_x], d.y[idx_y], d.rho[idx_x]);
            }
            fprintf(file, "\n");
        }
    }
    else {
        for (int j = 0; j < p.ny ; j++) {
            for (int i = 0; i < p.nx; i++) {
                int idx_x = index_1D(p, i, j);
                int idx_y = p.transpose_dims ? index_1D(p, j, i) : idx_x;
                fprintf(file, "%f, %f, %f\n", d.x[idx_x], d.y[idx_y], d.rho[idx_x]);
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);

    if (p.verbose < 2) {
        printf("\nWrote to file: %s", p.output_file);
    }
}


double time_loop(Params& p, Data& d)
{
    int cycles = 0;
    flt_t t = 0., dta = 0., dt;

    auto time_loop_start = std::chrono::steady_clock::now();

    auto&& [last_i, x, u, mask] = update_axis_parameters(p, d, p.current_axis);

    while (t < p.max_time && cycles < p.max_cycles) {
        TIC(); dt = dtCFL(p, d, dta);    TAC("dtCFL");

        if (!std::isfinite(dt) || dt <= 0.) {
            printf("Invalid dt at cycle %d: %f\n", cycles, dt);
            Kokkos::finalize();
            exit(1);
        }

        for (auto [axis, dt_factor] : split_axes(p, cycles)) {
            std::tie(last_i, x, u, mask) = update_axis_parameters(p, d, p.current_axis);

            TIC(); boundaryConditions(p, d);                         TAC("boundaryConditions");
            TIC(); numericalFluxes(p, d, dt * dt_factor, last_i, u); TAC("numericalFluxes");
            TIC(); cellUpdate(p, d, dt * dt_factor, u, x, mask);     TAC("cellUpdate");

            if (p.euler_projection) {
                TIC(); first_order_euler_remap(p, d, dt);            TAC("first_order_euler");
                if (p.transpose_dims)
                    transpose_parameters(p);
            }

            TIC(); update_EOS(p, d);                                 TAC("update_EOS");
        }

        dta = dt;
        cycles++;
        t += dt;

        if (p.verbose <= 1) {
            printf("Cycle = %4d, dt = %.18f, t = %.18f\n", cycles, dt, t);
        }
    }

    auto time_loop_end = std::chrono::steady_clock::now();

    double loop_time = std::chrono::duration<double>(time_loop_end - time_loop_start).count();
    double grind_time = loop_time / (static_cast<double>(cycles) * p.nb_cells) * 1e6;

    printf("\n");
    printf("Time:       %.4g seconds\n", loop_time);
    printf("Grind time: %.4g µs/cell/cycle\n", grind_time);
    printf("Cells/sec:  %.4g Mega cells/sec\n", 1. / grind_time);
    printf("Cycles:     %d\n", cycles);
    printf("Final dt:   %.18f\n\n", dt);

    return grind_time;
}


#ifdef KOKKOS_ENABLE_CUDA

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size(const Policy& policy)
{
    // See Kokkos_Cuda_Parallel.hpp : ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::Cuda>::execute()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i){};
    using FunctorType = decltype(functor);
    using LaunchBounds = typename Policy::launch_bounds;
    using ParallelFor = Kokkos::Impl::ParallelFor<FunctorType, Policy>;

    cudaFuncAttributes attr = Kokkos::Impl::CudaParallelLaunch<ParallelFor, LaunchBounds>::get_cuda_func_attributes();
    const unsigned block_size = Kokkos::Impl::cuda_get_opt_block_size<typename ParallelFor::functor_type, LaunchBounds>(policy.space().impl_internal_space_instance(), attr, functor, 1, 0, 0);

    auto max_grid = Kokkos::Impl::CudaInternal::singleton().m_maxBlock;

    dim3d block{1, block_size, 1};
    dim3d grid{
           std::min(
                   typename Policy::index_type((nb_work + block[1] - 1) / block[1]),
                   typename Policy::index_type(max_grid[0])),
           1, 1};

    return std::make_tuple(block, grid);
}

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size_reduction(const Policy& policy)
{
    // See Kokkos_Cuda_Parallel.hpp : ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType, Kokkos::Cuda>::local_block_size()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i, flt_t& reducer){};
    using FunctorType = decltype(functor);
    using ParallelReduce = Kokkos::Impl::ParallelReduce<FunctorType, Policy, flt_t, Kokkos::Cuda>;
    using return_value_adapter = Kokkos::Impl::ParallelReduceReturnValue<void, flt_t, FunctorType>;

    flt_t return_value{};
    Kokkos::Impl::ParallelReduce<FunctorType, Policy, typename return_value_adapter::reducer_type>
            reduction(functor, policy, return_value_adapter::return_value(return_value, functor));
    const unsigned block_size = reduction.local_block_size(functor);

    dim3d block{1, block_size, 1};
    dim3d grid{std::min(block[1], nb_work), 1, 1};

    return std::make_tuple(block, grid);
}

#else
#ifdef KOKKOS_ENABLE_HIP

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size(const Policy& policy)
{
    // See Kokkos_HIP_Parallel_Range.hpp : ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::Experimental::HIP>::execute()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i){};
    using FunctorType = decltype(functor);
    using LaunchBounds = typename Policy::launch_bounds;
    using ParallelFor = Kokkos::Impl::ParallelFor<FunctorType, Policy>;

    const unsigned block_size = Kokkos::Experimental::Impl::hip_get_preferred_blocksize<ParallelFor, LaunchBounds>();
    const dim3d block{1, block_size, 1};
    const dim3d grid{typename Policy::index_type((nb_work + block[1] - 1) / block[1]), 1, 1};

    return std::make_tuple(block, grid);
}

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size_reduction(const Policy& policy)
{
    // See Kokkos_HIP_Parallel_Range.hpp : ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType, Kokkos::Experimental::HIP>::local_block_size()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i, flt_t& reducer){};
    using FunctorType = decltype(functor);
    using ParallelReduce = Kokkos::Impl::ParallelReduce<FunctorType, Policy, flt_t, Kokkos::Experimental::HIP>;
    using return_value_adapter = Kokkos::Impl::ParallelReduceReturnValue<void, flt_t, FunctorType>;

    flt_t return_value{};
    Kokkos::Impl::ParallelReduce<FunctorType, Policy, typename return_value_adapter::reducer_type>
            reduction(functor, policy, return_value_adapter::return_value(return_value, functor));
    const unsigned block_size = reduction.local_block_size(functor);

    dim3d block{1, block_size, 1};
    dim3d grid{std::min(block[1], typename Policy::index_type((nb_work + block[1] - 1) / block[1])), 1, 1};

    return std::make_tuple(block, grid);
}

#else

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size(const Policy& policy)
{ return std::make_tuple(dim3d{1, 1, 1}, dim3d{1, 1, 1}); }

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size_reduction(const Policy& policy)
{ return std::make_tuple(dim3d{1, 1, 1}, dim3d{1, 1, 1}); }

#endif // KOKKOS_ENABLE_HIP
#endif // KOKKOS_ENABLE_CUDA


void print_kernel_params(const Params& p)
{
    auto [block, grid] = get_block_and_grid_size(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1));
    printf("Kernel launch parameters for 'parallel_for', with range [%d, %d]:\n", p.ideb, p.ifin + 1);
    printf(" - block dim: %d, %d, %d\n", block[0], block[1], block[2]);
    printf(" - grid dim:  %d, %d, %d\n", grid[0], grid[1], grid[2]);

    auto [block_r, grid_r] = get_block_and_grid_size_reduction(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1));
    printf("Kernel launch parameters for 'parallel_reduce', with range [%d, %d]:\n", p.ideb, p.ifin + 1);
    printf(" - block dim: %d, %d, %d\n", block_r[0], block_r[1], block_r[2]);
    printf(" - grid dim:  %d, %d, %d\n", grid_r[0], grid_r[1], grid_r[2]);
}


const char USAGE[] = R"(
 == Armon ==
CFD 1D solver using the conservative Euler equations in the lagrangian description.
Parallelized using Kokkos

Options:
    -h or --help            Prints this message and exit
    -t <test>               Test case: 'Sod', 'Sod_y', 'Sod_circ' or 'Bizarrium'
    -s <scheme>             Numeric scheme: 'Godunov' (first order) or 'GAD-minmod' (second order, minmod limiter)
    --cells Nx,Ny           Number of cells in the 2D mesh
    --cycle N               Maximum number of iterations
    --riemann <solver>      Riemann solver: 'acoustic' only
    --euler 0-1             Enable the eulerian projection step after each iteration
    --time T                Maximum time (in seconds)
    --cfl C                 CFL number
    --dt T                  Initial time step (in seconds)
    --cst-dt 0-1            Constant time step mode
    --write-output 0-1      If the variables should be written to the output file
    --write-ghosts 0-1      Include the ghost cells in the output file
    --output <file>         The output file name/path
    --write-throughput 0-1  Enable writing the cell throughput to a separate file (in Mega cells/sec)
    --transpose 0-1         If the axes should be transposed in order to optimize memory usage
    --splitting <method>    Axis splitting method: 'Sequential' (XYXY...), 'SequentialSym' (XYYXXYY...),
                            'Strang' (XYX with dt/2 for X, then YXY with dt/2 for Y, repeat)
    --verbose 0-3           Verbosity (0: high, 3: low)
)";


bool parse_arguments(Params& p, int argc, char** argv)
{
    errno = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) {
            if (strcmp(argv[i+1], "Sod") == 0) {
                p.test = Params::Test::Sod;
            }
            else if (strcmp(argv[i+1], "Sod_y") == 0) {
                p.test = Params::Test::Sod_y;
            }
            else if (strcmp(argv[i+1], "Sod_circ") == 0) {
                p.test = Params::Test::Sod_circ;
            }
            else if (strcmp(argv[i+1], "Bizarrium") == 0) {
                p.test = Params::Test::Bizarrium;
            }
            else {
                printf("Unknown test: %s\n", argv[i+1]);
                return false;
            }
            i++;
        }
        else if (strcmp(argv[i], "-s") == 0) {
            if (strcmp(argv[i+1], "Godunov") == 0) {
                p.scheme = Params::Scheme::Godunov;
            }
            else if (strcmp(argv[i+1], "GAD-minmod") == 0) {
                p.scheme = Params::Scheme::GAD_minmod;
            }
            else {
                printf("Wrong scheme: %s\n", argv[i+1]);
                return false;
            }
            i++;
        }
        else if (strcmp(argv[i], "--cells") == 0) {
            char* comma_pos = nullptr;
            p.nx = (int) strtol(argv[i+1], &comma_pos, 10);
            p.ny = (int) strtol(comma_pos+1, nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--cycle") == 0) {
            p.max_cycles = (int) strtol(argv[i+1], nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            p.verbose = (int) strtol(argv[i+1], nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--riemann") == 0) {
            if (strcmp(argv[i+1], "acoustic") == 0) {
                p.riemann = Params::Riemann::Acoustic;
            }
            else {
                printf("Wrong Riemann solver: %s\n", argv[i+1]);
                return false;
            }
            i++;
        }
        else if (strcmp(argv[i], "--time") == 0) {
            p.max_time = flt_t(strtod(argv[i+1], nullptr));
            i++;
        }
        else if (strcmp(argv[i], "--cfl") == 0) {
            p.cfl = flt_t(strtod(argv[i+1], nullptr));
            i++;
        }
        else if (strcmp(argv[i], "--dt") == 0) {
            p.Dt = flt_t(strtod(argv[i+1], nullptr));
            i++;
        }
        else if (strcmp(argv[i], "--write-output") == 0) {
            p.write_output = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--write-ghosts") == 0) {
            p.write_ghosts = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--euler") == 0) {
            p.euler_projection = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--cst-dt") == 0) {
            p.cst_dt = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--output") == 0) {
            p.output_file = argv[i+1];
            i++;
        }
        else if (strcmp(argv[i], "--write-throughput") == 0) {
            p.write_throughput = argv[i + 1];
            i++;
        }
        else if (strcmp(argv[i], "--transpose") == 0) {
            p.transpose_dims = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--splitting") == 0) {
            if (strcmp(argv[i+1], "Sequential") == 0) {
                p.axis_splitting = Params::AxisSplitting::Sequential;
            }
            else if (strcmp(argv[i+1], "SequentialSym") == 0) {
                p.axis_splitting = Params::AxisSplitting::SequentialSym;
            }
            else if (strcmp(argv[i+1], "Strang") == 0) {
                p.axis_splitting = Params::AxisSplitting::Strang;
            }
            else {
                printf("Wrong axis splitting method: %s\n", argv[i+1]);
                return false;
            }
            i++;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            puts(USAGE);
            return false;
        }
        else {
            fprintf(stderr, "Wrong option: %s\n", argv[i]);
            return false;
        }
    }

    if (errno != 0) {
        fprintf(stderr, "Parsing error occurred: %s\n", std::strerror(errno));
        return false;
    }

    return true;
}


bool check_parameters(Params& p)
{
    if (p.nx <= 0 || p.ny <= 0) {
        fputs("One of the dimensions of the domain is 0 or negative\n", stderr);
        return false;
    }

    if (p.cst_dt && p.Dt == 0.) {
        fputs("Constant time step is set ('--cst-dt 1') but the initial time step is 0\n", stderr);
        return false;
    }

    if (p.write_output && p.output_file == nullptr) {
        fputs("Write output is on but no output file was given\n", stderr);
        return false;
    }

    return true;
}


bool armon(int argc, char** argv)
{
    Params params;
    if (!parse_arguments(params, argc, argv)) return false;
    params.init_indexing();
    if (!check_parameters(params)) return false;

    if (params.verbose < 3) {
        params.print();
        print_kernel_params(params);
    }

    Data data("Armon", params.nb_cells + params.nb_ghosts * 2);
    HostData host_data = data.as_mirror();

    TIC(); init_test(params, data); TAC("init_test");
    double grind_time = time_loop(params, data);

    data.deep_copy_to_mirror(host_data);

    if (params.write_output) {
        write_output(params, host_data);
    }

    if (params.write_throughput) {
        FILE* grind_time_file = fopen("cell_throughput", "w");
        fprintf(grind_time_file, "%f", 1. / grind_time);
        fclose(grind_time_file);
    }

    if (params.verbose < 3) {
        double total_time = time_contribution["TOTAL"];
        time_contribution.erase(time_contribution.find("TOTAL"));

        printf("Total time for each step:\n");
        for (const auto& [label, time] : time_contribution) {
            printf(" - %-20s %10.5f ms (%5.2f%%)\n", label.c_str(), time * 1e3, time / total_time * 100);
        }
    }

    return true;
}


int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    bool ok = armon(argc, argv);
    Kokkos::finalize();
    return !ok;
}
