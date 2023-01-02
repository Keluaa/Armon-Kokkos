
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


std::unique_ptr<Kokkos::ScopeGuard> global_guard = nullptr;
void init_kokkos()
{
    if (global_guard == nullptr) {
        global_guard = std::make_unique<Kokkos::ScopeGuard>();
    }
}


template<>
struct doctest::StringMaker<std::tuple<int, int>>
{
    static String convert(const std::tuple<int, int>& t) {
        std::ostringstream oss;
        oss << "(" << std::get<0>(t) << ", " << std::get<1>(t) << ")";
        return oss.str().c_str();
    }
};


bool check_flt_eq(const std::string& msg, flt_t expected_value, flt_t value, flt_t tol = 1e-13)
{
    bool eq = is_approx(expected_value, value, tol);
    CHECK(eq);
    if (!eq) {
        std::streamsize tmp_precision = std::cout.precision();
        std::cout.precision(std::numeric_limits<flt_t>::digits10);
        std::cout << msg
                  << " Expected: " << expected_value
                  << ", got " << value
                  << " (diff: " << std::abs(expected_value - value) << ")\n";
        std::cout.precision(tmp_precision);
    }
    return eq;
}


Params get_default_params()
{
    Params p;

    p.test = Test::Sod;
    p.scheme = Scheme::GAD;
    p.limiter = Limiter::Minmod;
    p.projection = Projection::Euler_2nd;
    p.axis_splitting = AxisSplitting::Sequential;
    p.single_comm_per_axis_pass = false;

    p.max_cycles = 100;
    p.verbose = 1;

    p.nx = 100;
    p.ny = 5;
    p.nb_ghosts = 5;

    return p;
}


TEST_CASE("indexing") {
    Params p = get_default_params();
    p.single_comm_per_axis_pass = true;
    p.init();

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


TEST_CASE("Memory alignment") {
    init_kokkos();
    Params p = get_default_params();
    p.init();
    HostData data(p.nb_cells);
    for (auto& var : data.vars_array()) {
        auto ptr = reinterpret_cast<intptr_t>(var->data());
        CHECK_EQ(ptr % 64, 0);
    }
}


TEST_SUITE("NaNs check") {
    void run_nan_check(Test test_case)
    {
        init_kokkos();
        feenableexcept(FE_INVALID);

        Params ref_params = get_reference_params(test_case);
        Data data(ref_params.nb_cells);
        HostData host_data = data.as_mirror();
        init_test(ref_params, data);
        time_loop(ref_params, data, host_data);

        fedisableexcept(FE_INVALID);
    }


    TEST_CASE("Sod")       { run_nan_check(Test::Sod);       }
    TEST_CASE("Sod_y")     { run_nan_check(Test::Sod_y);     }
    TEST_CASE("Sod_circ")  { run_nan_check(Test::Sod_circ);  }
    TEST_CASE("Bizarrium") { run_nan_check(Test::Bizarrium); }
}


TEST_SUITE("Conservation") {
    void run_test(Test test_case)
    {
        init_kokkos();

        Params ref_params = get_reference_params(test_case);
        Data data(ref_params.nb_cells);
        HostData host_data = data.as_mirror();

        init_test(ref_params, data);
        auto [initial_mass, initial_energy] = conservation_vars(ref_params, data);
        time_loop(ref_params, data, host_data);
        auto [current_mass, current_energy] = conservation_vars(ref_params, data);

        check_flt_eq("Mass is not conserved.", initial_mass, current_mass, 1e-12);
        check_flt_eq("Energy is not conserved.", initial_energy, current_energy, 1e-12);
    }

    TEST_CASE("Sod")      { run_test(Test::Sod);      }
    TEST_CASE("Sod_y")    { run_test(Test::Sod_y);    }
    TEST_CASE("Sod_circ") { run_test(Test::Sod_circ); }
}


TEST_SUITE("Comparison with reference") {
    bool check_if_ref_file_exists(const std::string& path)
    {
        bool exists = std::filesystem::exists(path);
        if (!exists) {
            std::cerr << "Missing reference data file at: " << path << "\n";
        }
        return !exists;
    }


    void run_comparison(Test test, const std::string& ref_data_path)
    {
        init_kokkos();

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

        check_flt_eq("Final time steps are different.", ref_dt, dt);
        CHECK_EQ(cycles, ref_cycles);

        int diff_count = compare_with_reference(ref_params, ref_data, host_data);
        CHECK_EQ(diff_count, 0);
    }


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
