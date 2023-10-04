
#ifndef ARMON_KOKKOS_TEST_CASES_H
#define ARMON_KOKKOS_TEST_CASES_H

#include "common.h"


enum class Test : int {
    Sod = 0,
    Sod_y = 1,
    Sod_circ = 2,
    Bizarrium = 3,
    Sedov = 4
};


struct TestParams {
    flt_t high_rho, low_rho;
    flt_t high_E, low_E;
    flt_t high_u, low_u;
    flt_t high_v, low_v;
};


struct TestCase {
    virtual void init_params(TestParams& test_params) const = 0;
    virtual std::array<flt_t, 2> default_domain_size() const { return { 1, 1 }; }
    virtual std::array<flt_t, 2> default_domain_origin() const { return { 0, 0 }; }
    virtual flt_t default_CFL() const = 0;
    virtual flt_t default_max_time() const = 0;
    virtual std::array<flt_t, 2> boundaryCondition(Side side) const = 0;
    virtual flt_t test_option(flt_t, flt_t) const { return 0.; }
};


struct TestSod : TestCase {
    void init_params(TestParams& test_params) const override;
    flt_t default_CFL() const override { return 0.95; }
    flt_t default_max_time() const override { return 0.20; }
    std::array<flt_t, 2> boundaryCondition(Side side) const override
    {
        if (side == Side::Left || side == Side::Right)
            return { -1, 1 };
        else
            return { 1, 1 };
    }

    static KOKKOS_INLINE_FUNCTION
    bool region_high(flt_t x, flt_t) { return x <= 0.5; }
};


struct TestSodY : TestSod {
    std::array<flt_t, 2> boundaryCondition(Side side) const override
    {
        if (side == Side::Left || side == Side::Right)
            return { 1, 1 };
        else
            return { 1, -1 };
    }

    static KOKKOS_INLINE_FUNCTION
    bool region_high(flt_t, flt_t y) { return y <= 0.5; }
};


struct TestSodCirc : TestSod {
    std::array<flt_t, 2> boundaryCondition(Side side) const override
    {
        if (side == Side::Left || side == Side::Right)
            return { -1, 1 };
        else
            return { 1, -1 };
    }

    static KOKKOS_INLINE_FUNCTION
    bool region_high(flt_t x, flt_t y)
    {
        return (Kokkos::pow(x - flt_t(0.5), flt_t(2)) + Kokkos::pow(y - flt_t(0.5), flt_t(2))) <= flt_t(0.125);
    }
};


struct TestBizarrium : TestCase {
    void init_params(TestParams& test_params) const override;
    flt_t default_CFL() const override { return 0.6; }
    flt_t default_max_time() const override { return 80e-6; }
    std::array<flt_t, 2> boundaryCondition(Side side) const override
    {
        if (side == Side::Left || side == Side::Right)
            return { -1, 1 };
        else
            return { 1, 1 };
    }

    static KOKKOS_INLINE_FUNCTION
    bool region_high(flt_t x, flt_t) { return x <= 0.5; }
};


struct TestSedov : TestCase {
    flt_t r{};

    TestSedov() = default;
    explicit TestSedov(flt_t r) : r(r) {}

    void init_params(TestParams& test_params) const override;
    std::array<flt_t, 2> default_domain_size() const override { return { 2, 2 }; }
    std::array<flt_t, 2> default_domain_origin() const override { return { -1, -1 }; }
    flt_t default_CFL() const override { return 0.7; }
    flt_t default_max_time() const override { return 1.0; }
    std::array<flt_t, 2> boundaryCondition(Side side) const override
    {
        if (side == Side::Left || side == Side::Right)
            return { -1, 1 };
        else
            return { 1, 1 };
    }

    KOKKOS_INLINE_FUNCTION
    bool region_high(flt_t x, flt_t y) const
    {
        return (Kokkos::pow(x, flt_t(2)) + Kokkos::pow(y, flt_t(2))) <= Kokkos::pow(r, flt_t(2));
    }

    flt_t test_option(flt_t dx, flt_t dy) const override {
        return Kokkos::hypot(dx, dy) / Kokkos::numbers::sqrt2;
    }
};

#endif //ARMON_KOKKOS_TEST_CASES_H
