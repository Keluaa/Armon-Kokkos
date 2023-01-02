
#ifndef ARMON_KOKKOS_UTILS_H
#define ARMON_KOKKOS_UTILS_H

#include <cstdint>
#include <cmath>


#if __FAST_MATH__

// IEEE 754 compliant NAN and finiteness checks, which works even when fast math is enabled.
// Note: Kokkos seems to enable fast math even for debug builds.

static bool is_ieee754_nan(float x)
{
    constexpr uint32_t EXP_MASK = 0x7F800000;
    constexpr uint32_t SIGN_MASK = 0x80000000;
    uint32_t x_bits = reinterpret_cast<uint32_t&>(x);
    return (x_bits & EXP_MASK) == EXP_MASK && (x_bits & ~(EXP_MASK & SIGN_MASK)) != 0;
}


static bool is_ieee754_nan(double x)
{
    constexpr uint64_t EXP_MASK = 0x7FF0000000000000;
    constexpr uint64_t SIGN_MASK = 0x8000000000000000;
    uint64_t x_bits = reinterpret_cast<uint64_t&>(x);
    return (x_bits & EXP_MASK) == EXP_MASK && (x_bits & ~(EXP_MASK & SIGN_MASK)) != 0;
}


static bool is_ieee754_finite(float x)
{
    constexpr uint32_t EXP_MASK = 0x7F800000;
    uint32_t x_bits = reinterpret_cast<uint32_t&>(x);
    return (x_bits & EXP_MASK) != EXP_MASK;
}


static bool is_ieee754_finite(double x)
{
    constexpr uint64_t EXP_MASK = 0x7FF0000000000000;
    uint64_t x_bits = reinterpret_cast<uint64_t&>(x);
    return (x_bits & EXP_MASK) != EXP_MASK;
}

#else

static bool is_ieee754_nan(float x)
{
    return std::isnan(x);
}


static bool is_ieee754_nan(double x)
{
    return std::isnan(x);
}


static bool is_ieee754_finite(float x)
{
    return std::isfinite(x);
}


static bool is_ieee754_finite(double x)
{
    return std::isfinite(x);
}

#endif //__FAST_MATH__

#endif //ARMON_KOKKOS_UTILS_H
