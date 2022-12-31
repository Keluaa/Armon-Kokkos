
#include "reference.h"

#include "io.h"

#include <fstream>
#include <iomanip>


std::string get_reference_data_path(Test test_case)
{
    std::string ref_path = PROJECT_ROOT_DIR;
    ref_path += "/../julia/tests/reference_data/ref_";

    switch (test_case) {
    case Test::Sod:       ref_path += "Sod";       break;
    case Test::Sod_y:     ref_path += "Sod_y";     break;
    case Test::Sod_circ:  ref_path += "Sod_circ";  break;
    case Test::Bizarrium: ref_path += "Bizarrium"; break;
    }

#if USE_SINGLE_PRECISION
    ref_path += "_32bits.csv";
#else
    ref_path += "_64bits.csv";
#endif

    return ref_path;
}


Params get_reference_params(Test test_case)
{
    // Must be in sync with:
    // ../../julia/tests/reference_data/reference_functions.jl#get_reference_params
    Params params;

    params.test = test_case;
    params.scheme = Scheme::GAD;
    params.projection = Projection::Euler_2nd;
    params.limiter = Limiter::Minmod;
    params.axis_splitting = AxisSplitting::Sequential;
    params.single_comm_per_axis_pass = false;  // TODO: false for now since near perfect masking is required otherwise

    params.nb_ghosts = 5;
    params.nx = 100;
    params.ny = 100;

    params.max_cycles = 1000;
    params.max_time = 0;

    params.verbose = 5;
    params.write_output = false;

    params.init();

    return params;
}


std::tuple<HostData, flt_t, int> load_reference_data(const Params& ref_params, const std::string& ref_file_path)
{
    std::ifstream ref_file(ref_file_path);
    ref_file.exceptions(std::ios::failbit | std::ifstream::badbit);

    flt_t ref_dt;
    int ref_cycles;

    ref_file >> ref_dt;
    ref_file.ignore(1, ',');
    ref_file >> ref_cycles;

    HostData ref_data(ref_params.nb_cells);
    load_data(ref_params, ref_data, ref_file);

    return std::make_tuple(ref_data, ref_dt, ref_cycles);
}
