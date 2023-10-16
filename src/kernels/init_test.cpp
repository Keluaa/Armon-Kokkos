
#include <variant>

#include <Kokkos_Core.hpp>

#include "kernels.h"
#include "parallel_kernels.h"
#include "test_cases.h"
#include "utils.h"


void TestSod::init_params(TestParams& test_params) const
{
    test_params.high_rho = 1.;
    test_params.low_rho  = 0.125;
    test_params.high_E   = 2.5;
    test_params.low_E    = 2.0;
    test_params.high_u   = 0.;
    test_params.low_u    = 0.;
    test_params.high_v   = 0.;
    test_params.low_v    = 0.;
}


void TestBizarrium::init_params(TestParams& test_params) const
{
    test_params.high_rho = 1.42857142857e+4;
    test_params.low_rho  = 10000.;
    test_params.high_E   = 4.48657821135e+6;
    test_params.low_E    = 31250;
    test_params.high_u   = 0.;
    test_params.low_u    = 250.;
    test_params.high_v   = 0.;
    test_params.low_v    = 0.;
}


void TestSedov::init_params(TestParams& test_params) const
{
    test_params.high_rho = 1.;
    test_params.low_rho  = 1.;
    test_params.high_E   = 0.851072 / (M_PI * Kokkos::pow(r, flt_t(2)));
    test_params.low_E    = 2.5e-14;
    test_params.high_u   = 0.;
    test_params.low_u    = 0.;
    test_params.high_v   = 0.;
    test_params.low_v    = 0.;
}


template<typename TestCase>
void init_test(const Range& range, const InnerRange1D& inner_range,
               Idx row_length, Idx nb_ghosts,
               Idx nx, Idx ny, Idx g_nx, Idx g_ny, Idx pos_x, Idx pos_y,
               flt_t sx, flt_t sy, flt_t ox, flt_t oy,
               view& x, view& y, view& rho, view& Emat, view& umat, view& vmat,
               mask_view& domain_mask, view& pmat, view& cmat, view& gmat, view& ustar, view& pstar,
               TestParams test_params, TestCase test, bool debug_indexes)
{
    CHECK_VIEW_LABELS(x, y, rho, Emat, umat, vmat, domain_mask, pmat, cmat, gmat, ustar, pstar);
    parallel_kernel(range, KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);

        Idx ix = (i % row_length) - nb_ghosts;
        Idx iy = (i / row_length) - nb_ghosts;

        Idx g_ix = ix + pos_x;
        Idx g_iy = iy + pos_y;

        x[i] = flt_t(g_ix) / flt_t(nx) * sx + ox;
        y[i] = flt_t(g_iy) / flt_t(ny) * sy + oy;

        flt_t x_mid = x[i] + sx / flt_t(2 * g_nx);
        flt_t y_mid = y[i] + sy / flt_t(2 * g_ny);

        if (debug_indexes) {
            // +1 to compare with the Julia solver
            rho[i] = flt_t(i + 1);
            Emat[i] = flt_t(i + 1);
            umat[i] = flt_t(i + 1);
            vmat[i] = flt_t(i + 1);
        } else if (test.region_high(x_mid, y_mid)) {
            rho[i] = test_params.high_rho;
            Emat[i] = test_params.high_E;
            umat[i] = test_params.high_u;
            vmat[i] = test_params.high_v;
        } else {
            rho[i] = test_params.low_rho;
            Emat[i] = test_params.low_E;
            umat[i] = test_params.low_u;
            vmat[i] = test_params.low_v;
        }

        domain_mask[i] = (0 <= ix && ix < nx && 0 <= iy && iy < ny);

        // Set to zero to make sure no non-initialized values changes the result
        pmat[i] = 0;
        cmat[i] = 1;  // Set to 1 as a max speed of 0 will create NaNs
        gmat[i] = 0;
        ustar[i] = 0;
        pstar[i] = 0;
    });
}


extern "C"
void init_test(const Range& range, const InnerRange1D& inner_range,
               Idx row_length, Idx nb_ghosts, Idx nx, Idx ny, flt_t sx,
               flt_t sy, flt_t ox, flt_t oy, Idx pos_x, Idx pos_y, Idx g_nx, Idx g_ny,
               view& x, view& y, view& rho, view& Emat, view& umat, view& vmat,
               mask_view& domain_mask, view& pmat, view& cmat, view& gmat, view& ustar, view& pstar,
               Test test, bool debug_indexes, flt_t test_option)
KERNEL_TRY {
    std::variant<TestSod, TestSodY, TestSodCirc, TestBizarrium, TestSedov> test_case;
    switch (test) {
        case Test::Sod:       test_case = TestSod{};              break;
        case Test::Sod_y:     test_case = TestSodY{};             break;
        case Test::Sod_circ:  test_case = TestSodCirc{};          break;
        case Test::Bizarrium: test_case = TestBizarrium{};        break;
        case Test::Sedov:     test_case = TestSedov{test_option}; break;
        default:
            throw std::out_of_range(
                    "Invalid test case index: " + std::to_string(static_cast<int>(test))
                    + ", expected a value between " + std::to_string(static_cast<int>(Test::Sod))
                    + " and " + std::to_string(static_cast<int>(Test::Sedov))
            );
    }

    std::visit([&](auto&& test) {
        TestParams test_params{};
        test.init_params(test_params);
        init_test(
            range, inner_range,
            row_length, nb_ghosts,
            nx, ny, g_nx, g_ny, pos_x, pos_y,
            sx, sy, ox, oy,
            x, y, rho, Emat, umat, vmat,
            domain_mask, pmat, cmat, gmat, ustar, pstar,
            test_params, test, debug_indexes
        );
    }, test_case);
} KERNEL_CATCH
