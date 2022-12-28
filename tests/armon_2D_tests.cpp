
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "parameters.h"
#include "indexing.h"
#include "io.h"
#include "armon_2D.h"

#include "reference.h"

#include <filesystem>
#include <sstream>
#include <cfenv>


template<>
struct doctest::StringMaker<std::tuple<int, int>>
{
    static String convert(const std::tuple<int, int>& t) {
        std::ostringstream oss;
        oss << "(" << std::get<0>(t) << ", " << std::get<1>(t) << ")";
        return oss.str().c_str();
    }
};


TEST_CASE("indexing") {
    Params p;

    p.test = Test::Sod;
    p.scheme = Scheme::GAD;
    p.limiter = Limiter::Minmod;
    p.projection = Projection::Euler_2nd;
    p.axis_splitting = AxisSplitting::Sequential;
    p.single_comm_per_axis_pass = true;

    p.max_cycles = 100;
    p.verbose = 1;

    p.nx = 100;
    p.ny = 5;
    p.nb_ghosts = 5;

    p.init_indexing();

    SUBCASE("Domains") {
        CHECK_EQ(real_domain(p),           std::tuple<int, int>(333, 1316));
        CHECK_EQ(real_domain_fluxes(p),    std::tuple<int, int>(333, 1317));
        CHECK_EQ(real_domain_advection(p), std::tuple<int, int>(332, 1317));
        CHECK_EQ(all_cells(p),             std::tuple<int, int>(0, 1649));
    }

    SUBCASE("Indexes") {
        CHECK_EQ(index_1D(p, -1, 0), 554);
        CHECK_EQ(index_1D(p, 0, -1), 445);
        CHECK_EQ(index_1D(p, p.nx, 0), 655);
        CHECK_EQ(index_1D(p, 0, p.ny), 1105);
    }
}


void run_comparison(Test test, const std::string& ref_data_path)
{
    Params ref_params = get_reference_params(test);

    Data data(ref_params.nb_cells);
    HostData host_data = data.as_mirror();

    init_test(ref_params, data);
    flt_t dt;
    int cycles;
    std::tie(std::ignore, dt, cycles) = time_loop(ref_params, data, host_data);

    data.deep_copy_to_mirror(host_data);
    ref_params.output_file = "ref_output_test";
    write_output(ref_params, host_data);

    auto [ref_data, ref_dt, ref_cycles] = load_reference_data(ref_params, ref_data_path);

    CHECK(is_approx(dt, ref_dt));
    if (!is_approx(dt, ref_dt)) {
        std::streamsize tmp_precision = std::cout.precision();
        std::cout.precision(std::numeric_limits<flt_t>::digits10);
        std::cout << "Final time steps are different. Expected " << ref_dt << ", got " << dt << "\n";
        std::cout.precision(tmp_precision);
    }

    CHECK_EQ(cycles, ref_cycles);

    int diff_count = compare_with_reference(ref_params, ref_data, host_data);
    CHECK_EQ(diff_count, 0);
}


bool check_if_ref_file_exists(const std::string& path)
{
    bool exists = std::filesystem::exists(path);
    if (!exists) {
        std::cerr << "Missing reference data file at: " << path << "\n";
    }
    return !exists;
}


TEST_CASE("NaNs check") {
    feenableexcept(FE_INVALID);
    for (Test test_case : {Test::Sod, Test::Sod_y, Test::Sod_circ, Test::Bizarrium}) {
        Params ref_params = get_reference_params(test_case);
        Data data(ref_params.nb_cells);
        HostData host_data = data.as_mirror();
        init_test(ref_params, data);
        time_loop(ref_params, data, host_data);
    }
    fedisableexcept(FE_INVALID);
}


TEST_SUITE("Comparison with reference") {
    std::string ref_path_sod = get_reference_data_path(Test::Sod);
    bool sod_missing = check_if_ref_file_exists(ref_path_sod);
    TEST_CASE("Sod" * doctest::skip(sod_missing)) {
        run_comparison(Test::Sod, ref_path_sod);
    }

    std::string ref_path_sod_y = get_reference_data_path(Test::Sod_y);
    bool sod_y_missing = check_if_ref_file_exists(ref_path_sod_y);
    TEST_CASE("Sod_y" * doctest::skip(sod_y_missing)) {
        run_comparison(Test::Sod_y, ref_path_sod_y);
    }

    std::string ref_path_sod_circ = get_reference_data_path(Test::Sod_circ);
    bool sod_circ_missing = check_if_ref_file_exists(ref_path_sod_circ);
    TEST_CASE("Sod_circ" * doctest::skip(sod_circ_missing)) {
        // TODO: diverges from the correct solution when the wave reaches the borders (around cycle 16). NaNs occurs after.
        run_comparison(Test::Sod_circ, ref_path_sod_circ);
    }

    std::string ref_path_bizarrium = get_reference_data_path(Test::Bizarrium);
    bool bizarrium_missing = check_if_ref_file_exists(ref_path_bizarrium);
    TEST_CASE("Bizarrium" * doctest::skip(bizarrium_missing)) {
        run_comparison(Test::Bizarrium, ref_path_bizarrium);
    }
}
