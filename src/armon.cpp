
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
    enum class Test {
        Sod, Bizarrium
    } test = Test::Sod;

    enum class Scheme {
        Godunov, GAD_minmod
    } scheme = Scheme::GAD_minmod;

    enum class Riemann {
        Acoustic
    } riemann = Riemann::Acoustic;

    int nb_cells = 1000;
    int nb_ghosts = 2;
    int max_cycles = 100;

    int ideb = nb_ghosts;
    int ifin = nb_cells + nb_ghosts;

    flt_t max_time = 0.0;
    flt_t cfl = 0.6;
    flt_t dt = 0.0;
    flt_t Dt = 0.0;

    bool euler_projection = false;
    bool cst_dt = false;

    bool write_output = false;
    bool write_throughput = false;
    int verbose = 2;

    const char* output_file = "output_cpp";

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
#else
        printf(" - use gpu:    %d\n", 0);
#endif
        printf(" - ieee bits:  %lu\n", 8 * sizeof(flt_t));
        printf("\n");
        printf(" - test:       %s\n", (test == Test::Sod) ? "Sod" : "Bizarrium");
        printf(" - riemann:    %s\n", "acoustic");
        printf(" - scheme:     %s\n", (scheme == Scheme::Godunov) ? "Godunov" : "GAD-minmod");
        printf(" - nb cells:   %g\n", double(nb_cells));
        printf(" - CFL:        %g\n", cfl);
        printf(" - Dt init:    %g\n", Dt);
        printf(" - euler proj: %d\n", euler_projection);
        printf(" - cst dt:     %d\n", cst_dt);
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
    view_t x, X;
    view_t rho, umat, emat, Emat, pmat, cmat, gmat, ustar, pstar, ustar_1, pstar_1;
    view_t tmp_rho, tmp_urho, tmp_Erho;

    DataHolder() = default;

    [[maybe_unused]]
    DataHolder(const std::string& label, int size)
        : x(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , X(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , rho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , umat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , emat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
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
        , tmp_Erho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
    { }

    [[nodiscard]]
    DataHolder<typename view_t::HostMirror> as_mirror() const
    {
        DataHolder<typename view_t::HostMirror> mirror;
        mirror.x = Kokkos::create_mirror_view(x);
        mirror.X = Kokkos::create_mirror_view(X);
        mirror.rho = Kokkos::create_mirror_view(rho);
        mirror.umat = Kokkos::create_mirror_view(umat);
        mirror.emat = Kokkos::create_mirror_view(emat);
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
        mirror.tmp_Erho = Kokkos::create_mirror_view(tmp_Erho);
        return mirror;
    }

    template<typename mirror_view_t>
    void deep_copy_to_mirror(DataHolder<mirror_view_t>& mirror) const
    {
        Kokkos::deep_copy(mirror.x, x);
        Kokkos::deep_copy(mirror.X, X);
        Kokkos::deep_copy(mirror.rho, rho);
        Kokkos::deep_copy(mirror.umat, umat);
        Kokkos::deep_copy(mirror.emat, emat);
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
        Kokkos::deep_copy(mirror.tmp_Erho, tmp_Erho);
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


void perfectGasEOS(const Params& p, Data& d, flt_t gamma)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin),
    KOKKOS_LAMBDA(const int i){
        d.pmat[i] = (gamma - 1) * d.rho[i] * d.emat[i];
        d.cmat[i] = std::sqrt(gamma * d.pmat[i] / d.rho[i]);
        d.gmat[i] = (1 + gamma) / 2;
    });
}


void bizarriumEOS(const Params& p, Data& d)
{
    const flt_t rho0 = 1e4, K0 = 1e11, Cv0 = 1e3, T0 = 300, eps0 = 0;
    const flt_t G0 = 1.5, s = 1.5, q = -42080895./14941154., r = 727668333./149411540.;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin),
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

        d.pmat[i] = pk0 + G0*rho0*(d.emat[i] - eps_k0);
        d.cmat[i] = std::sqrt(G0*rho0*(d.pmat[i] - pk0) - pk0prime) / d.rho[i];
        d.gmat[i] = flt_t(0.5) / (std::pow(d.rho[i],flt_t(3)) * std::pow(d.cmat[i],flt_t(2)))
                * (pk0second + std::pow(G0 * rho0,flt_t(2)) * (d.pmat[i]-pk0));
    });
}


void init_test(Params& p, Data& d)
{
    switch (p.test) {
    case Params::Test::Sod:
    {
        if (p.max_time == 0.0) p.max_time = 0.20;
        if (p.cfl == 0.0) p.cfl = 0.95;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
        KOKKOS_LAMBDA(const int i) {
            d.x[i] = flt_t(i - p.nb_ghosts) / flt_t(p.nb_cells);
            if (d.x[i] < 0.5) {
                d.rho[i] = 1.;
                d.pmat[i] = 1.;
                d.umat[i] = 0.;
            }
            else {
                d.rho[i] = 0.125;
                d.pmat[i] = 0.1;
                d.umat[i] = 0.0;
            }

            const flt_t gamma = 1.4;
            d.emat[i] = d.Emat[i] = d.pmat[i] / ((gamma - 1) * d.rho[i]);
            d.cmat[i] = std::sqrt(gamma * d.pmat[i] / d.rho[i]);
            d.gmat[i] = flt_t(0.5) * (1 + gamma);
        });

        break;
    }
    case Params::Test::Bizarrium:
    {
        if (p.max_time == 0.0) p.max_time = 80e-6;
        if (p.cfl == 0.0) p.cfl = 0.6;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
        KOKKOS_LAMBDA(const int i) {
            d.x[i] = flt_t(i - p.nb_ghosts) / flt_t(p.nb_cells);
            if (d.x[i] < 0.5) {
                d.rho[i] = 1.42857142857e+4;
                d.pmat[i] = 0.;
                d.emat[i] = d.Emat[i] = 4.48657821135e+6;
            }
            else {
                d.rho[i] =  10000.;
                d.umat[i] = 250.;
                d.emat[i] = 0.;
                d.Emat[i] = d.emat[i] + flt_t(0.5) * std::pow(d.umat[i], flt_t(2));
            }
        });

        bizarriumEOS(p, d);

        break;
    }
    }
}


void boundaryConditions(const Params& p, Data& d)
{
    Kokkos::parallel_for(1,
    KOKKOS_LAMBDA(const int i) {
        d.rho[p.ideb-1]  = d.rho[p.ideb];    d.rho[p.ifin] = d.rho[p.ifin-1];
        d.umat[p.ideb-1] = -d.umat[p.ideb];
        d.pmat[p.ideb-1] = d.pmat[p.ideb];  d.pmat[p.ifin] = d.pmat[p.ifin-1];
        d.cmat[p.ideb-1] = d.cmat[p.ideb];  d.cmat[p.ifin] = d.cmat[p.ifin-1];
        d.gmat[p.ideb-1] = d.gmat[p.ideb];  d.gmat[p.ifin] = d.gmat[p.ifin-1];

        if (p.test == Params::Test::Bizarrium) {
            d.umat[p.ifin] = d.umat[p.ifin-1];
        }
        else {
            d.umat[p.ifin] = -d.umat[p.ifin-1];
        }
    });
}


void first_order_euler_remap(const Params& p, Data& d, flt_t dt)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
    KOKKOS_LAMBDA(const int i) {
        flt_t dx = d.X[i+1] - d.X[i];
        flt_t L1 =  std::max(flt_t(0.), d.ustar[i]) * dt;
        flt_t L3 = -std::min(flt_t(0.), d.ustar[i+1]) * dt;
        flt_t L2 = dx - L1 - L3;

        d.tmp_rho[i]  = (L1*d.rho[i-1]             + L2*d.rho[i]           + L3*d.rho[i+1]            ) / dx;
        d.tmp_urho[i] = (L1*d.rho[i-1]*d.umat[i-1] + L2*d.rho[i]*d.umat[i] + L3*d.rho[i+1]*d.umat[i+1]) / dx;
        d.tmp_Erho[i] = (L1*d.rho[i-1]*d.Emat[i-1] + L2*d.rho[i]*d.Emat[i] + L3*d.rho[i+1]*d.Emat[i+1]) / dx;
    });

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
    KOKKOS_LAMBDA(const int i) {
        d.rho[i]  = d.tmp_rho[i];
        d.umat[i] = d.tmp_urho[i] / d.tmp_rho[i];
        d.Emat[i] = d.tmp_Erho[i] / d.tmp_rho[i];
    });
}


flt_t dtCFL(const Params& p, Data& d, flt_t dta)
{
    flt_t dt = INFINITY;

    if (p.cst_dt) {
        return p.Dt;
    }
    else if (p.euler_projection) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(p.ideb, p.ifin),
        KOKKOS_LAMBDA(const int i, flt_t& dt_loop) {
            flt_t max_c = std::max(std::abs(d.umat[i] + d.cmat[i]), std::abs(d.umat[i] - d.cmat[i]));
            dt_loop = std::min(dt_loop, ((d.x[i+1] - d.x[i]) / max_c));
        }, Kokkos::Min<flt_t>(dt));
    }
    else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(p.ideb, p.ifin),
        KOKKOS_LAMBDA(const int i, flt_t& dt_loop) {
            dt_loop = std::min(dt_loop, ((d.x[i+1] - d.x[i]) / d.cmat[i]));
        }, Kokkos::Min<flt_t>(dt));
    }

    if (dta == 0) {
        if (p.Dt != 0) {
            return p.Dt;
        }
        else {
            return p.cfl * dt;
        }
    }
    else {
        return std::min(p.cfl * dt, flt_t(1.05) * dta);
    }
}


void acoustic(const Params& p, Data& d)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
    KOKKOS_LAMBDA(const int i){
        flt_t rc_l = d.rho[i-1] * d.cmat[i-1];
        flt_t rc_r = d.rho[i]   * d.cmat[i];

        d.ustar[i] = (rc_l * d.umat[i-1] + rc_r * d.umat[i] + (d.pmat[i-1] - d.pmat[i])) / (rc_l + rc_r);
        d.pstar[i] = (rc_r * d.pmat[i-1] + rc_l * d.pmat[i] + rc_l * rc_r * (d.umat[i-1] - d.umat[i])) / (rc_l + rc_r);
    });
}


KOKKOS_INLINE_FUNCTION flt_t phi(flt_t r)
{
    // Minmod
    return std::max(flt_t(0.), std::min(flt_t(1.), r));
}


void acoustic_GAD(const Params& p, Data& d, flt_t dt)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
    KOKKOS_LAMBDA(const int i) {
        flt_t rc_l = d.rho[i-1] * d.cmat[i-1];
        flt_t rc_r = d.rho[i]   * d.cmat[i];

        d.ustar_1[i] = (rc_l * d.umat[i-1] + rc_r * d.umat[i] + (d.pmat[i-1] - d.pmat[i])) / (rc_l + rc_r);
        d.pstar_1[i] = (rc_r * d.pmat[i-1] + rc_l * d.pmat[i] + rc_l * rc_r * (d.umat[i-1] - d.umat[i])) / (rc_l + rc_r);
    });

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
    KOKKOS_LAMBDA(const int i){
        flt_t r_u_m = (d.ustar_1[i+1] - d.umat[i]) / (d.ustar_1[i] - d.umat[i-1] + flt_t(1e-6));
        flt_t r_p_m = (d.pstar_1[i+1] - d.pmat[i]) / (d.pstar_1[i] - d.pmat[i-1] + flt_t(1e-6));
        flt_t r_u_p = (d.umat[i-1] - d.ustar_1[i-1]) / (d.umat[i] - d.ustar_1[i] + flt_t(1e-6));
        flt_t r_p_p = (d.pmat[i-1] - d.pstar_1[i-1]) / (d.pmat[i] - d.pstar_1[i] + flt_t(1e-6));
//        if (std::isnan(r_u_m)) r_u_m = 1;
//        if (std::isnan(r_p_m)) r_p_m = 1;
//        if (std::isnan(r_u_p)) r_u_p = 1;
//        if (std::isnan(r_p_p)) r_p_p = 1;

        flt_t dm_l = d.rho[i-1] * (d.x[i] - d.x[i-1]);
        flt_t dm_r = d.rho[i]   * (d.x[i+1] - d.x[i]);
        flt_t Dm = (dm_l + dm_r) / 2;
        flt_t theta = ((d.rho[i-1] * d.cmat[i-1]) + (d.rho[i] * d.cmat[i])) / 2 * (dt / Dm);

        d.ustar[i] = d.ustar_1[i] + flt_t(0.5) * (1 - theta)
                * (phi(r_u_p) * (d.umat[i] - d.ustar_1[i]) - phi(r_u_m) * (d.ustar_1[i] - d.umat[i-1]));
        d.pstar[i] = d.pstar_1[i] + flt_t(0.5) * (1 - theta)
                * (phi(r_p_p) * (d.pmat[i] - d.pstar_1[i]) - phi(r_p_m) * (d.pstar_1[i] - d.pmat[i-1]));
    });
}


void numericalFluxes(const Params& p, Data& d, flt_t dt)
{
    switch (p.riemann) {
    case Params::Riemann::Acoustic:
    {
        switch (p.scheme) {
        case Params::Scheme::Godunov:    acoustic(p, d); break;
        case Params::Scheme::GAD_minmod: acoustic_GAD(p, d, dt); break;
        }
        break;
    }
    }
}


void cellUpdate(const Params& p, Data& d, flt_t dt)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
    KOKKOS_LAMBDA(const int i){
        d.X[i] = d.x[i] + dt * d.ustar[i];
    });

    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin),
    KOKKOS_LAMBDA(const int i){
        flt_t dm = d.rho[i] * (d.x[i+1] - d.x[i]);
        d.rho[i] = dm / (d.X[i+1] - d.X[i]);
        d.umat[i] += dt / dm * (d.pstar[i] - d.pstar[i+1]);
        d.Emat[i] += dt / dm * (d.pstar[i] * d.ustar[i] - d.pstar[i+1] * d.ustar[i+1]);
        d.emat[i] = d.Emat[i] - flt_t(0.5) * std::pow(d.umat[i], flt_t(2));
    });

    if (!p.euler_projection) {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
        KOKKOS_LAMBDA(const int i){
            d.x[i] = d.X[i];
        });
    }
}


void update_EOS(const Params& p, Data& d)
{
    switch (p.test) {
    case Params::Test::Sod:
    {
        const flt_t gamma = 1.4;
        perfectGasEOS(p, d, gamma);
        break;
    }
    case Params::Test::Bizarrium: bizarriumEOS(p, d); break;
    }
}


double time_loop(const Params& p, Data& d)
{
    int cycles = 0;
    flt_t t = 0., dta = 0.;

    auto time_loop_start = std::chrono::steady_clock::now();

    while (t < p.max_time && cycles < p.max_cycles) {
        TIC(); boundaryConditions(p, d);    TAC("boundaryConditions");
        TIC(); flt_t dt = dtCFL(p, d, dta); TAC("dtCFL");
        TIC(); numericalFluxes(p, d, dt);   TAC("numericalFluxes");
        TIC(); cellUpdate(p, d, dt);        TAC("cellUpdate");

        if (p.euler_projection) {
            TIC(); first_order_euler_remap(p, d, dt); TAC("first_order_euler");
        }

        TIC(); update_EOS(p, d);            TAC("update_EOS");

        dta = dt;
        cycles++;
        t += dt;

        if (p.verbose <= 1) {
            printf("Cycle = %d, dt = %.3g, t = %.3g\n", cycles, dt, t);
        }

        if (!std::isfinite(dt) || dt <= .0) {
            printf("Error: dt has an invalid value: %f\n", dt);
            Kokkos::finalize();
            exit(1);
        }
    }

    auto time_loop_end = std::chrono::steady_clock::now();

    double loop_time = std::chrono::duration<double>(time_loop_end - time_loop_start).count();
    double grind_time = loop_time / (static_cast<double>(cycles) * p.nb_cells) * 1e6;

    printf("\n");
    printf("Time:       %.4g seconds\n", loop_time);
    printf("Grind time: %.4g µs/cell/cycle\n", grind_time);
    printf("Cells/sec:  %.4g Mega cells/sec\n", 1. / grind_time);
    printf("Cycles:     %d\n\n", cycles);

    return grind_time;
}


void write_output(const Params& p, const HostData& d)
{
    printf("Writing to output file '%s'...\n", p.output_file);

    FILE* file = fopen(p.output_file, "w");
    for (int i = p.ideb; i < p.ifin; i++) {
        fprintf(file, "%f, %f, %f, %f, %f, %f, %f\n",
                (d.x[i] + d.x[i+1]) * 0.5, d.rho[i], d.umat[i], d.pmat[i], d.emat[i], d.cmat[i], d.gmat[i]);
    }
    fclose(file);

    printf("Done.\n\n");
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
    auto [block, grid] = get_block_and_grid_size(Kokkos::RangePolicy<>(p.ideb, p.ifin));
    printf("Kernel launch parameters for 'parallel_for', with range [%d, %d]:\n", p.ideb, p.ifin);
    printf(" - block dim: %d, %d, %d\n", block[0], block[1], block[2]);
    printf(" - grid dim:  %d, %d, %d\n", grid[0], grid[1], grid[2]);

    auto [block_r, grid_r] = get_block_and_grid_size_reduction(Kokkos::RangePolicy<>(p.ideb, p.ifin));
    printf("Kernel launch parameters for 'parallel_reduce', with range [%d, %d]:\n", p.ideb, p.ifin);
    printf(" - block dim: %d, %d, %d\n", block_r[0], block_r[1], block_r[2]);
    printf(" - grid dim:  %d, %d, %d\n", grid_r[0], grid_r[1], grid_r[2]);
}


const char USAGE[] = R"(
 == Armon ==
CFD 1D solver using the conservative Euler equations in the lagrangian description.
Parallelized using Kokkos

Options:
    -h or --help            Prints this message and exit
    -t <test>               Test case: 'Sod' or 'Bizarrium'
    -s <scheme>             Numeric scheme: 'Godunov' (first order) or 'GAD-minmod' (second order, minmod limiter)
    --cells N               Number of cells in the mesh
    --cycle N               Maximum number of iterations
    --riemann <solver>      Riemann solver: 'acoustic' only
    --euler 0-1             Enable the eulerian projection step after each iteration
    --time T                Maximum time (in seconds)
    --cfl C                 CFL number
    --dt T                  Initial time step (in seconds)
    --cst-dt 0-1            Constant time step mode
    --write-output 0-1      If the variables should be written to the output file
    --output <file>         The output file name/path
    --write-throughput 0-1  Enable writing the cell throughput to a separate file (in Mega cells/sec)
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
            p.nb_cells = (int) strtol(argv[i+1], nullptr, 10);
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
    if (p.cst_dt && p.Dt == 0.) {
        fputs("Constant time step is set ('--cst-dt 1') but the initial time step is 0", stderr);
        return false;
    }

    if (p.write_output && p.output_file == nullptr) {
        fputs("Write output is on but no output file was given", stderr);
        return false;
    }

    return true;
}


bool armon(int argc, char** argv)
{
    Params params;
    if (!parse_arguments(params, argc, argv)) return false;
    if (!check_parameters(params)) return false;

    params.ideb = params.nb_ghosts;
    params.ifin = params.nb_ghosts + params.nb_cells;

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
