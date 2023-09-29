
#ifndef ARMON_KOKKOS_LIMITERS_H
#define ARMON_KOKKOS_LIMITERS_H

#include <Kokkos_Core.hpp>


enum class Limiter : int {
    None = 0,
    Minmod = 1,
    Superbee = 2
};


template <Limiter L>
KOKKOS_INLINE_FUNCTION flt_t limiter(flt_t)
{
    static_assert(L == Limiter::None, "Wrong limiter type");
    return flt_t(1);
}


template<>
KOKKOS_INLINE_FUNCTION flt_t limiter<Limiter::Minmod>(flt_t r)
{
    return Kokkos::max(flt_t(0), Kokkos::min(flt_t(1), r));
}


template<>
KOKKOS_INLINE_FUNCTION flt_t limiter<Limiter::Superbee>(flt_t r)
{
    return Kokkos::max(Kokkos::max(flt_t(0), Kokkos::min(flt_t(1), 2*r)), Kokkos::min(flt_t(2), r));
}

#endif //ARMON_KOKKOS_LIMITERS_H
