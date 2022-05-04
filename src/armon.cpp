
#include <cstring>
#include <cstdio>
#include <cstdlib>

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
            const char* num_threads_raw = getenv("KOKKOS_NUM_THREADS");
            int num_threads = 1;
            if (num_threads_raw != nullptr) {
                num_threads = (int) strtol(num_threads_raw, nullptr, 10);
            }
            printf(" - multithreading: 1, (%d)\n", num_threads);
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


void perfectGasEOS(const Params& p, Data& d, double gamma)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin),
    KOKKOS_LAMBDA(const int i){
        d.pmat[i] = (gamma - 1) * d.rho[i] * d.emat[i];
        d.cmat[i] = std::sqrt(gamma * d.pmat[i] / d.rho[i]);
        d.gmat[i] = (1 + gamma) / 2;
    });
}


void init_test(Params& p, Data& d)
{
    switch (p.test) {
    case Params::Test::Sod:
    {
        if (p.max_time == 0.0) p.max_time = 0.20;
        if (p.cfl == 0.0) p.cfl = 0.95;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(p.ideb, p.ifin),
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
            d.pmat[i] = (gamma - 1) * d.rho[i] * d.emat[i];
            d.cmat[i] = std::sqrt(gamma * d.pmat[i] / d.rho[i]);
            d.gmat[i] = (1 + gamma) / 2;
        });

        break;
    }
    case Params::Test::Bizarrium:
    {
        // TODO
        break;
    }
    }
}


void write_output(const Params& p, const HostData& d)
{
    printf("Writing to output file...\n");

    FILE* file = fopen("output_cpp", "w");
    for (int i = p.ideb; i < p.ifin; i++) {
        fprintf(file, "%f, %f, %f, %f, %f, %f, %f\n",
                (d.x[i] + d.x[i+1]) * 0.5, d.rho[i], d.umat[i], d.pmat[i], d.emat[i], d.cmat[i], d.gmat[i]);
    }
    fclose(file);

    printf("Done.\n");
}


bool parse_arguments(Params& p, char** argv)
{
    errno = 0;

    while (*argv != nullptr) {
        if (strcmp(*argv, "-t") == 0) {
            argv++;
            if (strcmp(*argv, "Sod") == 0) {
                p.test = Params::Test::Sod;
            }
            else if (strcmp(*argv, "Bizarrium") == 0) {
                p.test = Params::Test::Bizarrium;
            }
            else {
                printf("Unknown test: %s\n", *argv);
                return false;
            }
        }
        else if (strcmp(*argv, "-s") == 0) {
            argv++;
            if (strcmp(*argv, "Godunov") == 0) {
                p.scheme = Params::Scheme::Godunov;
            }
            else if (strcmp(*argv, "GAD-minmod") == 0) {
                p.scheme = Params::Scheme::GAD_minmod;
            }
            else {
                printf("Wrong scheme: %s\n", *argv);
                return false;
            }
        }
        else if (strcmp(*argv, "--cells") == 0) {
            argv++;
            p.nb_cells = (int) strtol(*argv, nullptr, 10);
        }
        else if (strcmp(*argv, "--cycle") == 0) {
            argv++;
            p.max_cycles = (int) strtol(*argv, nullptr, 10);
        }
        else if (strcmp(*argv, "--verbose") == 0) {
            argv++;
            p.verbose = (int) strtol(*argv, nullptr, 10);
        }
        else if (strcmp(*argv, "--riemann") == 0) {
            argv++;
            if (strcmp(*argv, "acoustic") == 0) {
                p.riemann = Params::Riemann::Acoustic;
            }
            else {
                printf("Wrong Riemann solver: %s\n", *argv);
            }
        }
        else if (strcmp(*argv, "--time") == 0) {
            argv++;
            p.max_time = strtod(*argv, nullptr);
        }
        else if (strcmp(*argv, "--cfl") == 0) {
            argv++;
            p.cfl = strtod(*argv, nullptr);
        }
        else if (strcmp(*argv, "--dt") == 0) {
            argv++;
            p.Dt = strtod(*argv, nullptr);
        }
        else if (strcmp(*argv, "--write-output") == 0) {
            argv++;
            p.write_output = strtol(*argv, nullptr, 2);
        }
        else if (strcmp(*argv, "--euler") == 0) {
            argv++;
            p.euler_projection = strtol(*argv, nullptr, 2);
        }
        else if (strcmp(*argv, "--cst-dt") == 0) {
            argv++;
            p.cst_dt = strtol(*argv, nullptr, 2);
        }
        else if (strcmp(*argv, "--output") == 0) {
            argv++;
            p.output_file = *argv;
        }
        else {
            printf("Wrong option: %s\n", *argv);
            return false;
        }

        argv++;
    }

    if (errno != 0) {
        printf("Parsing error occurred: %s\n", std::strerror(errno));
        return false;
    }

    return true;
}


void armon(char** argv)
{
    Params params;
    if (!parse_arguments(params, argv)) return;

    params.ideb = params.nb_ghosts;
    params.ifin = params.nb_ghosts + params.nb_cells;

    if (params.verbose < 3) params.print();

    Data data("Armon", params.nb_cells + params.nb_ghosts * 2);
    HostData host_data = data.as_mirror();

    init_test(params, data);

    data.deep_copy_to_mirror(host_data);

    if (params.write_output) {
        write_output(params, host_data);
    }
}


int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    Kokkos::print_configuration(std::cout);

    armon(argv + 1);  // ignore the first argument, since it is the path to the executable

    Kokkos::finalize();
}
