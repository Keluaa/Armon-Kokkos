
#include <cstring>

#include "parameters.h"
#include "data.h"
#include "armon_2D.h"


const char USAGE[] = R"(
 == Armon ==
CFD 1D solver using the conservative Euler equations in the lagrangian description.
Parallelized using Kokkos.

Options:
    -h or --help            Prints this message and exit
    -t <test>               Test case: 'Sod', 'Sod_y', 'Sod_circ', 'Bizarrium' or 'Sedov'
    -s <scheme>             Numeric scheme: 'Godunov' (first order) or 'GAD' (second order)
    --limiter <limiter>     Limiter for the second order scheme: 'None', 'Minmod' (default), 'Superbee'
    --cells Nx,Ny           Number of cells in the 2D mesh
    --nghost N              Number of ghost cells around the 2D domain
    --stencil N             Width of the stencil in cells. Defaults to the minimum number of ghost cells.
    --cycle N               Maximum number of iterations
    --riemann <solver>      Riemann solver: 'acoustic' only
    --projection <scheme>   Projection scheme: 'euler' (1st order) or 'euler_2nd' (2nd order)
    --time T                Maximum simulation time (in seconds)
    --cfl C                 CFL number
    --dt T                  Initial time step (in seconds)
    --cst-dt 0-1            Constant time step mode
    --write-output 0-1      If the variables should be written to the output file
    --write-ghosts 0-1      Include the ghost cells in the output file
    --output <file>         The output file name/path
    --precision D           Number of digits of precision printed to the output file for each value.
    --write-throughput 0-1  Enable writing the cell throughput to a separate file (in Mega cells/sec)
    --splitting <method>    Axis splitting method: 'Sequential' (XYXY...), 'SequentialSym' (XYYXXYY...),
                            'Strang' (XYX with dt/2 for X, then YXY with dt/2 for Y, repeat)
    --compare 0-1           Compare each step of the solver with some reference data.
    --verbose 0-3           Verbosity (0: high, 3: low)
)";


bool parse_arguments(Params& p, int argc, char** argv)
{
    errno = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) {
            if (strcmp(argv[i+1], "Sod") == 0) {
                p.test = Test::Sod;
            }
            else if (strcmp(argv[i+1], "Sod_y") == 0) {
                p.test = Test::Sod_y;
            }
            else if (strcmp(argv[i+1], "Sod_circ") == 0) {
                p.test = Test::Sod_circ;
            }
            else if (strcmp(argv[i+1], "Bizarrium") == 0) {
                p.test = Test::Bizarrium;
            }
            else if (strcmp(argv[i+1], "Sedov") == 0) {
                p.test = Test::Sedov;
            }
            else {
                printf("Unknown test: %s\n", argv[i+1]);
                return false;
            }
            i++;
        }
        else if (strcmp(argv[i], "-s") == 0) {
            if (strcmp(argv[i+1], "Godunov") == 0) {
                p.scheme = Scheme::Godunov;
            }
            else if (strcmp(argv[i+1], "GAD") == 0) {
                p.scheme = Scheme::GAD;
            }
            else {
                printf("Wrong scheme: %s\n", argv[i+1]);
                return false;
            }
            i++;
        }
        else if (strcmp(argv[i], "--limiter") == 0) {
            if (strcmp(argv[i+1], "None") == 0) {
                p.limiter = Limiter::None;
            }
            else if (strcmp(argv[i+1], "Minmod") == 0) {
                p.limiter = Limiter::Minmod;
            }
            else if (strcmp(argv[i+1], "Superbee") == 0) {
                p.limiter = Limiter::Superbee;
            }
            else {
                printf("Wrong limiter: %s\n", argv[i+1]);
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
        else if (strcmp(argv[i], "--nghost") == 0) {
            p.nb_ghosts = (int) strtol(argv[i+1], nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--stencil") == 0) {
            p.stencil_width = (int) strtol(argv[i+1], nullptr, 10);
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
                p.riemann = Riemann::Acoustic;
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
        else if (strcmp(argv[i], "--precision") == 0) {
            p.output_precision = (int) strtol(argv[i+1], nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--projection") == 0) {
            if (strcmp(argv[i+1], "none") == 0) {
                p.projection = Projection::None;
            } else if (strcmp(argv[i+1], "euler") == 0) {
                p.projection = Projection::Euler;
            } else if (strcmp(argv[i+1], "euler_2nd") == 0) {
                p.projection = Projection::Euler_2nd;
            } else {
                printf("Wrong projection scheme: %s\n", argv[i+1]);
                return false;
            }
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
            p.write_throughput = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--splitting") == 0) {
            if (strcmp(argv[i+1], "Sequential") == 0) {
                p.axis_splitting = AxisSplitting::Sequential;
            }
            else if (strcmp(argv[i+1], "SequentialSym") == 0) {
                p.axis_splitting = AxisSplitting::SequentialSym;
            }
            else if (strcmp(argv[i+1], "Strang") == 0) {
                p.axis_splitting = AxisSplitting::Strang;
            }
            else if (strcmp(argv[i+1], "X_only") == 0) {
                p.axis_splitting = AxisSplitting::X_only;
            }
            else if (strcmp(argv[i+1], "Y_only") == 0) {
                p.axis_splitting = AxisSplitting::Y_only;
            }
            else {
                printf("Wrong axis splitting method: %s\n", argv[i+1]);
                return false;
            }
            i++;
        }
        else if (strcmp(argv[i], "--compare") == 0) {
            p.compare = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--comparison-tolerance") == 0) {
            p.comparison_tolerance = strtod(argv[i+1], nullptr);
            i++;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            puts(USAGE);
            Kokkos::finalize();
            exit(0);
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


bool run_armon(int argc, char* argv[])
{
    Params params;
    if (!parse_arguments(params, argc, argv)) return false;
    params.init();
    if (!params.check()) return false;

    if (params.verbose < 3) {
        params.print();
        Kokkos::print_configuration(std::cout, true);
    }

    return armon(params);
}


int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    bool ok = run_armon(argc, argv);
    Kokkos::finalize();
    return !ok;
}
