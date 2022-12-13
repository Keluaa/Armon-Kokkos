
#include "reference.h"

#include "indexing.h"

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

    params.max_cycles = 100;
    params.max_time = 1;

    params.verbose = 5;
    params.write_output = false;

    params.init_indexing();

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

    HostData ref_data("ArmonRefTest", ref_params.nb_cells);

    const std::array vars = {&ref_data.x, &ref_data.y, &ref_data.rho, &ref_data.umat, &ref_data.vmat, &ref_data.pmat};

    for (int j = 0; j < ref_params.ny; j++) {
        for (int i = 0; i < ref_params.ny; i++) {
            int idx = index_1D(ref_params, i, j);

            auto it = vars.cbegin();
            ref_file >> (*it)->operator[](idx);
            for (it++; it != vars.cend(); it++) {
                ref_file.ignore(1, ',');
                ref_file >> (*it)->operator[](idx);
            }
        }
    }

    return std::make_tuple(ref_data, ref_dt, ref_cycles);
}


bool is_approx(flt_t a, flt_t b)
{
    return std::abs(a - b) <= flt_t(1e-13);
}


int compare_with_reference(const Params& ref_params, const HostData& ref_data, const HostData& data)
{
    const std::array field_names = {"x", "y", "rho", "umat", "vmat", "pmat"};
    const std::array ref_vars = {&ref_data.x, &ref_data.y, &ref_data.rho, &ref_data.umat, &ref_data.vmat, &ref_data.pmat};
    const std::array vars     = {&    data.x, &    data.y, &    data.rho, &    data.umat, &    data.vmat, &    data.pmat};

    int total_differences_count = 0;

    for (int j = 0; j < ref_params.ny; j++) {
        int row_deb = index_1D(ref_params, 0, j);
        int row_fin = index_1D(ref_params, ref_params.nx, j);

        auto ref_it = ref_vars.cbegin(), it = vars.cbegin();
        for (int field_i = 0; field_i < ref_vars.size(); field_i++, ref_it++, it++) {
            int row_diff_count = 0;
            flt_t max_diff = 0;
            for (int idx = row_deb; idx < row_fin; idx++) {
                flt_t ref_val = (*ref_it)->operator[](idx);
                flt_t val = (*ref_it)->operator[](idx);
                row_diff_count += !is_approx(ref_val, val);
                max_diff = std::max(max_diff, std::abs(ref_val - val));
            }

            total_differences_count += row_diff_count;

            if (row_diff_count > 0) {
                std::streamsize tmp_precision = std::cout.precision();
                std::cout.precision(std::numeric_limits<flt_t>::digits10);
                std::cout << "Row " << (j+1) << " has " << row_diff_count
                          << " differences (max diff: " << max_diff << ")"
                          << " in '" << field_names[field_i]
                          << "' with the reference\n";
                std::cout.precision(tmp_precision);
            }
        }
    }

    return total_differences_count;
}
