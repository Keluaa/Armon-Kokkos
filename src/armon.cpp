
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <string>

#include <cfenv>

#include <Kokkos_Core.hpp>

#define ON true
#define OFF false

#ifndef USE_GPU
#define USE_GPU 0
#endif

#ifndef USE_THREADING
#define USE_THREADING 0
#endif


typedef double flt_t;

using view = Kokkos::View<flt_t*>;
using host_view = view::HostMirror;


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
    int verbose = 2;

    const char* output_file = "output_cpp";

    void print() const
    {
        printf("Parameters:\n");
        if (USE_THREADING) {
            int max_num_threads = Kokkos::DefaultExecutionSpace::concurrency();
            printf(" - multithreading: 1, (%d threads)\n", max_num_threads);
        }
        else {
            printf(" - multithreading: 0\n");
        }
        printf(" - use simd:   TODO\n"); // TODO
        printf(" - use gpu:    %d\n", USE_GPU);
        printf(" - ieee bits:  %lu\n", 8 * sizeof(flt_t));
        printf("\n");
        printf(" - test:       %s\n", (test == Test::Sod) ? "Sod" : "Bizarrium");
        printf(" - riemann:    %s\n", "acoustic");
        printf(" - scheme:     %s\n", (scheme == Scheme::Godunov) ? "Godunov" : "GAD-minmod");
        printf(" - nb cells:   %d\n", nb_cells);
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

    DataHolder(const std::string& label, int size)
        : x(label, size)
        , X(label, size)
        , rho(label, size)
        , umat(label, size)
        , emat(label, size)
        , Emat(label, size)
        , pmat(label, size)
        , cmat(label, size)
        , gmat(label, size)
        , ustar(label, size)
        , pstar(label, size)
        , ustar_1(label, size)
        , pstar_1(label, size)
        , tmp_rho(label, size)
        , tmp_urho(label, size)
        , tmp_Erho(label, size)
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

        flt_t eps_k0 = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*(x*x)*f0;
        flt_t pk0 = -Cv0*T0*G0*rho0 + 0.5*K0*x*pow(1+x,2)*(2*f0+x*f1);
        flt_t pk0prime = -0.5*K0*pow(1+x,3)*rho0 * (2*(1+3*x)*f0 + 2*x*(2+3*x)*f1 + (x*x)*(1+x)*f2);
        flt_t pk0second = 0.5*K0*pow(1+x,4)*(rho0*rho0)
                            *(12*(1+2*x)*f0 + 6*(1+6*x+6*(x*x))*f1 + 6*x*(1+x)*(1+2*x)*f2 + pow(x*(1+x),2)*f3);

        d.pmat[i] = pk0 + G0*rho0*(d.emat[i] - eps_k0);
        d.cmat[i] = sqrt(G0*rho0*(d.pmat[i] - pk0) - pk0prime) / d.rho[i];

        d.gmat[i] = 0.5/(pow(d.rho[i],3)*pow(d.cmat[i],2))*(pk0second+pow(G0*rho0,2)*(d.pmat[i]-pk0));
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
            d.x[i] = static_cast<flt_t>(i - p.nb_ghosts) / p.nb_cells;
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
            d.cmat[i] = sqrt(gamma * d.pmat[i] / d.rho[i]);
            d.gmat[i] = 0.5 * (1 + gamma);
        });

        break;
    }
    case Params::Test::Bizarrium:
    {
        if (p.max_time == 0.0) p.max_time = 80e-6;
        if (p.cfl == 0.0) p.cfl = 0.6;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin + 1),
        KOKKOS_LAMBDA(const int i) {
            d.x[i] = static_cast<flt_t>(i - p.nb_ghosts) / p.nb_cells;
            if (d.x[i] < 0.5) {
                d.rho[i] = 1.42857142857e+4;
                d.pmat[i] = 0.;
                d.emat[i] = d.Emat[i] = 4.48657821135e+6;
            }
            else {
                d.rho[i] =  10000.;
                d.umat[i] = 250.;
                d.emat[i] = 0.;
                d.Emat[i] = d.emat[i] + 0.5 * pow(d.umat[i], 2);
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
        flt_t L1 =  std::max(0., d.ustar[i]) * dt;
        flt_t L3 = -std::min(0., d.ustar[i+1]) * dt;
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
        return std::min(p.cfl * dt, 1.05 * dta);
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
    return std::max(0., std::min(1., r));
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
        flt_t r_u_m = (d.ustar_1[i+1] - d.umat[i]) / (d.ustar_1[i] - d.umat[i-1] + 1e-6);
        flt_t r_p_m = (d.pstar_1[i+1] - d.pmat[i]) / (d.pstar_1[i] - d.pmat[i-1] + 1e-6);
        flt_t r_u_p = (d.umat[i-1] - d.ustar_1[i-1]) / (d.umat[i] - d.ustar_1[i] + 1e-6);
        flt_t r_p_p = (d.pmat[i-1] - d.pstar_1[i-1]) / (d.pmat[i] - d.pstar_1[i] + 1e-6);
//        if (std::isnan(r_u_m)) r_u_m = 1;
//        if (std::isnan(r_p_m)) r_p_m = 1;
//        if (std::isnan(r_u_p)) r_u_p = 1;
//        if (std::isnan(r_p_p)) r_p_p = 1;

        flt_t dm_l = d.rho[i-1] * (d.x[i] - d.x[i-1]);
        flt_t dm_r = d.rho[i]   * (d.x[i+1] - d.x[i]);
        flt_t Dm = (dm_l + dm_r) / 2;
        flt_t theta = ((d.rho[i-1] * d.cmat[i-1]) + (d.rho[i] * d.cmat[i])) / 2 * (dt / Dm);

        d.ustar[i] = d.ustar_1[i] + 0.5 * (1 - theta)
                * (phi(r_u_p) * (d.umat[i] - d.ustar_1[i]) - phi(r_u_m) * (d.ustar_1[i] - d.umat[i-1]));
        d.pstar[i] = d.pstar_1[i] + 0.5 * (1 - theta)
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
        d.emat[i] = d.Emat[i] - 0.5 * std::pow(d.umat[i], 2);
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


void time_loop(const Params& p, Data& d)
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
    }

    auto time_loop_end = std::chrono::steady_clock::now();

    double loop_time = std::chrono::duration<double>(time_loop_end - time_loop_start).count();
    double grind_time = loop_time / (static_cast<double>(cycles) * p.nb_cells) * 1e6;

    printf("\n");
    printf("Time:       %.4g seconds\n", loop_time);
    printf("Grind time: %.4g µs/cell/cycle\n", grind_time);
    printf("Cells/sec:  %.4g Mega cells/sec\n", 1. / grind_time);
    printf("Cycles:     %d\n\n", cycles);
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
            p.max_time = strtod(argv[i+1], nullptr);
            i++;
        }
        else if (strcmp(argv[i], "--cfl") == 0) {
            p.cfl = strtod(argv[i+1], nullptr);
            i++;
        }
        else if (strcmp(argv[i], "--dt") == 0) {
            p.Dt = strtod(argv[i+1], nullptr);
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

    if (params.verbose < 3) params.print();

    Data data("Armon", params.nb_cells + params.nb_ghosts * 2);
    HostData host_data = data.as_mirror();

    TIC(); init_test(params, data); TAC("init_test");
    time_loop(params, data);

    data.deep_copy_to_mirror(host_data);

    if (params.write_output) {
        write_output(params, host_data);
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
    feenableexcept(FE_INVALID);

    Kokkos::initialize(argc, argv);

    Kokkos::print_configuration(std::cout);

    bool ok = armon(argc, argv);

    Kokkos::finalize();

    return !ok;
}
