
#ifndef ARMON_KOKKOS_UTILS_H
#define ARMON_KOKKOS_UTILS_H

#include "common.h"

#include <string>


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

#define APPLY_4(f, arg1, arg2, arg3, arg4, ...)   \
    APPLY_3(f, arg1, arg2, arg3, ## __VA_ARGS__); \
    APPLY_1(f, arg4, ## __VA_ARGS__)

#define APPLY_5(f, arg1, arg2, arg3, arg4, arg5, ...) \
    APPLY_3(f, arg1, arg2, arg3, ## __VA_ARGS__);     \
    APPLY_2(f, arg4, arg5, ## __VA_ARGS__)

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
    APPLY_4(f, arg1, arg2, arg3, arg4, ## __VA_ARGS__);                       \
    APPLY_4(f, arg5, arg6, arg7, arg8, ## __VA_ARGS__);                       \
    APPLY_1(f, arg9, ## __VA_ARGS__)

#define APPLY_10(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, ...) \
    APPLY_4(f, arg1, arg2, arg3, arg4, ## __VA_ARGS__);                               \
    APPLY_4(f, arg5, arg6, arg7, arg8, ## __VA_ARGS__);                               \
    APPLY_2(f, arg9, arg10, ## __VA_ARGS__)

#define APPLY_11(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, ...) \
    APPLY_4(f, arg1, arg2, arg3, arg4, ## __VA_ARGS__);                                      \
    APPLY_4(f, arg5, arg6, arg7, arg8, ## __VA_ARGS__);                                      \
    APPLY_3(f, arg9, arg10, arg11, ## __VA_ARGS__)

#define APPLY_12(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, ...) \
    APPLY_4(f, arg1, arg2, arg3, arg4, ## __VA_ARGS__);                                             \
    APPLY_4(f, arg5, arg6, arg7, arg8, ## __VA_ARGS__);                                             \
    APPLY_4(f, arg9, arg10, arg11, arg12, ## __VA_ARGS__)


#define VA_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, N, ...) N
#define VA_NARGS(...) VA_NARGS_IMPL(__VA_ARGS__, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)


#define DYN_APPLY_IMPL_2(expr) expr
#define DYN_APPLY_IMPL_1(prefix, suffix) DYN_APPLY_IMPL_2(prefix ## suffix)
#define DYN_APPLY(prefix, suffix) DYN_APPLY_IMPL_1(prefix, suffix)


#ifdef CHECK_VIEW_ORDER
inline bool ends_with(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

template<typename View_t>
inline void check_view_label(const View_t& view, const std::string& suffix, const std::string_view& function)
{
    if (!ends_with(view.label(), suffix)) {
        std::ostringstream oss;
        oss << "Wrong view order in " << function << ": "
            << "expected " << suffix << ", got: " << view.label() << "\n";
        throw std::invalid_argument(oss.str());
    }
}

#define CHECK_VIEW_LABEL_IMPL(view) check_view_label(view, #view, __PRETTY_FUNCTION__)
#define CHECK_VIEW_LABELS_IMPL(APPLY_MACRO, ...) APPLY_MACRO(CHECK_VIEW_LABEL_IMPL, __VA_ARGS__)
#define CHECK_VIEW_LABELS(...) CHECK_VIEW_LABELS_IMPL(DYN_APPLY(APPLY_, VA_NARGS(__VA_ARGS__)), __VA_ARGS__)
#else
#define CHECK_VIEW_LABELS(...) void()
#endif //CHECK_VIEW_ORDER


extern "C" DLL_EXPORT
void (*raise_exception_handler)(const char* kernel, const char* msg) __attribute__((noreturn));


#ifdef TRY_ALL_CALLS
__attribute__((noreturn)) void raise_exception(const char* kernel_name, const std::exception& exception);

#define KERNEL_TRY try
#define KERNEL_CATCH                    \
    catch (const std::exception& err) { \
        raise_exception(__func__, err); \
    }
#else
#define KERNEL_TRY
#define KERNEL_CATCH
#endif //TRY_ALL_CALLS


/**
 * Unpack 'expr' into 'a' and 'b', but as constant values, not as structured bindings.
 *
 * Structured bindings are incompatible with lambda captures (until C++20) and OpenMP, and avoiding them is verbose,
 * hence this macro.
 */
#define CONST_UNPACK(a, b, expr)        \
    const auto [_ ## a, _ ## b] = expr; \
    const auto a = _ ## a;              \
    const auto b = _ ## b


#endif //ARMON_KOKKOS_UTILS_H
