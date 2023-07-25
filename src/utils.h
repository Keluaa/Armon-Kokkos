
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


#define APPLY_IMPL(expr) expr

#define APPLY_1(f, arg1, ...) \
    APPLY_IMPL(f(arg1, ## __VA_ARGS__))

#define APPLY_2(f, arg1, arg2, ...)      \
    APPLY_IMPL(f(arg1, ## __VA_ARGS__)); \
    APPLY_IMPL(f(arg2, ## __VA_ARGS__))

#define APPLY_3(f, arg1, arg2, arg3, ...) \
    APPLY_IMPL(f(arg1, ## __VA_ARGS__));  \
    APPLY_IMPL(f(arg2, ## __VA_ARGS__));  \
    APPLY_IMPL(f(arg3, ## __VA_ARGS__))

#define APPLY_4(f, arg1, arg2, arg3, arg4, ...) \
    APPLY_IMPL(f(arg1, ## __VA_ARGS__));        \
    APPLY_IMPL(f(arg2, ## __VA_ARGS__));        \
    APPLY_IMPL(f(arg3, ## __VA_ARGS__));        \
    APPLY_IMPL(f(arg4, ## __VA_ARGS__))

#define APPLY_5(f, arg1, arg2, arg3, arg4, arg5, ...) \
    APPLY_IMPL(f(arg1, ## __VA_ARGS__));              \
    APPLY_IMPL(f(arg2, ## __VA_ARGS__));              \
    APPLY_IMPL(f(arg3, ## __VA_ARGS__));              \
    APPLY_IMPL(f(arg4, ## __VA_ARGS__));              \
    APPLY_IMPL(f(arg5, ## __VA_ARGS__))

#define APPLY_6(f, arg1, arg2, arg3, arg4, arg5, arg6, ...) \
    APPLY_3(f, arg1, arg2, arg3, ## __VA_ARGS__);           \
    APPLY_3(f, arg4, arg5, arg6, ## __VA_ARGS__)

#define APPLY_7(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, ...) \
    APPLY_4(f, arg1, arg2, arg3, arg4, ## __VA_ARGS__);           \
    APPLY_3(f, arg5, arg6, arg7, ## __VA_ARGS__)

#define APPLY_8(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, ...) \
    APPLY_4(f, arg1, arg2, arg3, arg4, ## __VA_ARGS__);                 \
    APPLY_4(f, arg5, arg6, arg7, arg8, ## __VA_ARGS__)

#define APPLY_9(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, ...) \
    APPLY_5(f, arg1, arg2, arg3, arg4, arg5, ## __VA_ARGS__);                 \
    APPLY_4(f, arg6, arg7, arg8, arg9, ## __VA_ARGS__)

#define APPLY_10(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, ...) \
    APPLY_5(f, arg1, arg2, arg3, arg4, arg5, ## __VA_ARGS__);                         \
    APPLY_5(f, arg6, arg7, arg8, arg9, arg10, ## __VA_ARGS__)

#define APPLY_11(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, ...) \
    APPLY_6(f, arg1, arg2, arg3, arg4, arg5, arg6, ## __VA_ARGS__);                          \
    APPLY_5(f, arg7, arg8, arg9, arg10, arg11, ## __VA_ARGS__)

#define APPLY_12(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, ...) \
    APPLY_6(f, arg1, arg2, arg3, arg4, arg5, arg6, ## __VA_ARGS__);                                 \
    APPLY_6(f, arg7, arg8, arg9, arg10, arg11, arg12, ## __VA_ARGS__)

#define VA_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, N, ...) N
#define VA_NARGS(...) VA_NARGS_IMPL(__VA_ARGS__, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define DYN_APPLY_IMPL_2(expr) expr
#define DYN_APPLY_IMPL_1(prefix, n) DYN_APPLY_IMPL_2(prefix ## n)
#define DYN_APPLY(prefix, n) DYN_APPLY_IMPL_1(prefix, n)

#define UNPACK_FIELD_IMPL(FIELD, STRUCT) auto FIELD = STRUCT.FIELD
#define UNPACK_FIELDS_IMPL(APPLY_MACRO, STRUCT, ...) APPLY_MACRO(UNPACK_FIELD_IMPL, __VA_ARGS__, STRUCT)
#define UNPACK_FIELDS(STRUCT, ...) UNPACK_FIELDS_IMPL(DYN_APPLY(APPLY_, VA_NARGS(__VA_ARGS__)), STRUCT, __VA_ARGS__)

#endif //ARMON_KOKKOS_UTILS_H
