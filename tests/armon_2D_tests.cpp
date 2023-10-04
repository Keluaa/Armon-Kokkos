
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "parameters.h"
#include "kernels/indexing.h"
#include "io.h"
#include "armon_2D.h"
#include "ranges.h"
#include "utils.h"

#include "reference.h"

#include <filesystem>
#include <sstream>
#include <cfenv>
#include <csignal>
#include <csetjmp>


std::unique_ptr<Kokkos::ScopeGuard> global_guard = nullptr;
void init_kokkos()
{
    if (global_guard == nullptr) {
        global_guard = std::make_unique<Kokkos::ScopeGuard>();
    }
}


template<>
struct doctest::StringMaker<DomainRange>
{
    static String convert(const DomainRange& dr)
    {
        std::ostringstream oss;
        oss << "DomainRange{" << dr.col_start << ":" << dr.col_step << ":" << dr.col_end
            << ", " << dr.row_start << ":" << dr.row_step << ":" << dr.row_end << "}";
        return oss.str().c_str();
    }
};


bool check_flt_eq(const std::string& msg, flt_t expected_value, flt_t value, flt_t tol = 1e-13)
{
    CHECK(is_ieee754_finite(value));
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

    p.max_cycles = 100;
    p.verbose = 1;

    // Rectangular domain to catch edge-cases
    p.nx = 100;
    p.ny = 5;
    p.nb_ghosts = 5;

    return p;
}


TEST_CASE("indexing") {
    Params p = get_default_params();
    p.init();
    REQUIRE(p.check());

    SUBCASE("Domains") {
        CHECK_EQ(real_domain(p),      DomainRange{555, 110,  995,  0, 1,  99});
        CHECK_EQ(domain_fluxes(p),    DomainRange{555, 110,  995, -2, 1, 102});
        CHECK_EQ(domain_advection(p), DomainRange{555, 110,  995,  0, 1, 100});
        CHECK_EQ(complete_domain(p),  DomainRange{  5, 110, 1545, -5, 1, 104});

        CHECK_EQ(real_domain(p).length(),       500);
        CHECK_EQ(domain_fluxes(p).length(),     525);
        CHECK_EQ(domain_advection(p).length(),  505);
        CHECK_EQ(complete_domain(p).length(),  1650);

        CHECK_EQ(real_domain(p).row_length(),      100);
        CHECK_EQ(domain_fluxes(p).row_length(),    105);
        CHECK_EQ(domain_advection(p).row_length(), 101);
        CHECK_EQ(complete_domain(p).row_length(),  110);

        CHECK_EQ(real_domain(p).col_length(),       5);
        CHECK_EQ(domain_fluxes(p).col_length(),     5);
        CHECK_EQ(domain_advection(p).col_length(),  5);
        CHECK_EQ(complete_domain(p).col_length(),  15);

        CHECK_EQ(real_domain(p).begin(),      555);
        CHECK_EQ(domain_fluxes(p).begin(),    553);
        CHECK_EQ(domain_advection(p).begin(), 555);
        CHECK_EQ(complete_domain(p).begin(),    0);

        CHECK_EQ(real_domain(p).end(),      1094);
        CHECK_EQ(domain_fluxes(p).end(),    1097);
        CHECK_EQ(domain_advection(p).end(), 1095);
        CHECK_EQ(complete_domain(p).end(),  1649);
    }

    SUBCASE("Ranges") {
        SUBCASE("1D") {
            auto domain = real_domain(p);
            auto [range, inner_range] = domain.iter1D();

            CHECK_EQ(range.start,      0);
            CHECK_EQ(range.end,      540);
            CHECK_EQ(range.length(), 540);

            CHECK_EQ(inner_range.start, 555);
            CHECK_EQ(inner_range.scale_index(range.start), domain.begin());
            CHECK_EQ(inner_range.scale_index(range.end-1),   domain.end());
        }

        SUBCASE("2D") {
            auto domain = real_domain(p);
            auto [range, inner_range] = domain.iter2D();

            CHECK_EQ(range.start,      0);
            CHECK_EQ(range.end,      500);
            CHECK_EQ(range.length(), 500);

            CHECK_EQ(inner_range.main_range_start, 555);
            CHECK_EQ(inner_range.main_range_step,  110);
            CHECK_EQ(inner_range.row_range_start,    0);
            CHECK_EQ(inner_range.row_range_length, 100);

            CHECK_EQ(inner_range.scale_index(range.start), domain.begin());
            CHECK_EQ(inner_range.scale_index(range.end-1),   domain.end());  // range.end is exclusive, domain.end() is inclusive => -1
            CHECK_EQ(inner_range.scale_index(range.start +   inner_range.row_range_length), domain.begin() +   p.row_length);
            CHECK_EQ(inner_range.scale_index(range.start + 2*inner_range.row_range_length), domain.begin() + 2*p.row_length);
        }
    }

    SUBCASE("Boundary Conditions") {
        int disp;
        DomainRange dr;

        dr = boundary_conditions_domain(p, Side::Bottom, disp);
        CHECK_EQ(dr.begin(),  445);
        CHECK_EQ(dr.end(),    544);
        CHECK_EQ(dr.length(), 100);
        CHECK_EQ(disp,        110);

        dr = boundary_conditions_domain(p, Side::Left, disp);
        CHECK_EQ(dr.begin(),  554);
        CHECK_EQ(dr.end(),    994);
        CHECK_EQ(dr.length(),   5);
        CHECK_EQ(disp,          1);

        dr = boundary_conditions_domain(p, Side::Right, disp);
        CHECK_EQ(dr.begin(),  655);
        CHECK_EQ(dr.end(),   1095);
        CHECK_EQ(dr.length(),   5);
        CHECK_EQ(disp,         -1);

        dr = boundary_conditions_domain(p, Side::Top, disp);
        CHECK_EQ(dr.begin(), 1105);
        CHECK_EQ(dr.end(),   1204);
        CHECK_EQ(dr.length(), 100);
        CHECK_EQ(disp,       -110);
    }

    SUBCASE("Indexes") {
        CHECK_EQ(index_1D(p, -1, 0),    554);
        CHECK_EQ(index_1D(p, 0, -1),    445);
        CHECK_EQ(index_1D(p, p.nx, 0),  655);
        CHECK_EQ(index_1D(p, 0, p.ny), 1105);
    }
}


TEST_CASE("Memory alignment") {
    init_kokkos();
    Params p = get_default_params();
    p.init();
    REQUIRE(p.check());
    HostData data(p.nb_cells);
    for (auto& var : data.vars_array()) {
        auto ptr = reinterpret_cast<intptr_t>(var->data());
        CHECK_EQ(ptr % 64, 0);
    }
}


TEST_SUITE("NaNs check") {
    std::jmp_buf jump_buffer;

    void sigfpe_handler(int)
    {
        std::longjmp(jump_buffer, 1);
    }

    void run_nan_check(Test test_case, int expected_cycles)
    {
        init_kokkos();
        feenableexcept(FE_INVALID);
        std::signal(SIGFPE, sigfpe_handler);

        Params ref_params = get_reference_params(test_case);
        REQUIRE(ref_params.check());
        Data data(ref_params.nb_cells);
        HostData host_data = data.as_mirror();
        volatile int cycles = 0;

        if (setjmp(jump_buffer) == 0) {
            init_test(ref_params, data);
            cycles = std::get<2>(time_loop(ref_params, data, host_data));
        } else {
            FAIL("Invalid floating-point operation");
        }

        fedisableexcept(FE_INVALID);
        std::signal(SIGFPE, SIG_DFL);

        CHECK_EQ(cycles, expected_cycles);
    }

    TEST_CASE("Sod")       { run_nan_check(Test::Sod,       45); }
    TEST_CASE("Sod_y")     { run_nan_check(Test::Sod_y,     45); }
    TEST_CASE("Sod_circ")  { run_nan_check(Test::Sod_circ,  44); }
    TEST_CASE("Bizarrium") { run_nan_check(Test::Bizarrium, 76); }
    TEST_CASE("Sedov")     { run_nan_check(Test::Sedov,    568); }
}


TEST_SUITE("Conservation") {
    void run_test(Test test_case, int expected_cycles)
    {
        init_kokkos();

        Params ref_params = get_reference_params(test_case);
        REQUIRE(ref_params.check());
        Data data(ref_params.nb_cells);
        HostData host_data = data.as_mirror();

        init_test(ref_params, data);
        auto [initial_mass, initial_energy] = conservation_vars(ref_params, data);
        int cycles = std::get<2>(time_loop(ref_params, data, host_data));
        auto [current_mass, current_energy] = conservation_vars(ref_params, data);

        check_flt_eq("Mass is not conserved.", initial_mass, current_mass, 1e-12);
        check_flt_eq("Energy is not conserved.", initial_energy, current_energy, 1e-12);

        CHECK_EQ(cycles, expected_cycles);
    }

    TEST_CASE("Sod")      { run_test(Test::Sod,      45); }
    TEST_CASE("Sod_y")    { run_test(Test::Sod_y,    45); }
    TEST_CASE("Sod_circ") { run_test(Test::Sod_circ, 44); }
    // Bizarrium is not conservative
    TEST_CASE("Sedov")    { run_test(Test::Sedov,   568); }
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
        REQUIRE(ref_params.check());

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
    TEST_CASE("ref_Sod" * doctest::skip(sod_missing)) {
        run_comparison(Test::Sod, ref_path_sod);
    }

    std::string ref_path_sod_y = get_reference_data_path(Test::Sod_y);
    bool sod_y_missing = check_if_ref_file_exists(ref_path_sod_y);
    TEST_CASE("ref_Sod_y" * doctest::skip(sod_y_missing)) {
        run_comparison(Test::Sod_y, ref_path_sod_y);
    }

    std::string ref_path_sod_circ = get_reference_data_path(Test::Sod_circ);
    bool sod_circ_missing = check_if_ref_file_exists(ref_path_sod_circ);
    TEST_CASE("ref_Sod_circ" * doctest::skip(sod_circ_missing)) {
        // TODO: diverges from the correct solution
        run_comparison(Test::Sod_circ, ref_path_sod_circ);
    }

    std::string ref_path_bizarrium = get_reference_data_path(Test::Bizarrium);
    bool bizarrium_missing = check_if_ref_file_exists(ref_path_bizarrium);
    TEST_CASE("ref_Bizarrium" * doctest::skip(bizarrium_missing)) {
        // TODO: diverges from the correct solution
        // NOTE: GCC rounds the pressure in the initial call to 'update_EOS' differently from Clang, by a single ulp.
        run_comparison(Test::Bizarrium, ref_path_bizarrium);
    }

    std::string ref_path_sedov = get_reference_data_path(Test::Sedov);
    bool sedov_missing = check_if_ref_file_exists(ref_path_sedov);
    TEST_CASE("ref_Sedov" * doctest::skip(sedov_missing)) {
        // TODO: diverges from the correct solution
        // NOTE: GCC rounds the pressure in the initial call to 'update_EOS' differently from Clang, by a single ulp.
        run_comparison(Test::Sedov, ref_path_sedov);
    }
}
