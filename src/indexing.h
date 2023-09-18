
#ifndef ARMON_KOKKOS_INDEXING_H
#define ARMON_KOKKOS_INDEXING_H

#include "Kokkos_Core.hpp"

#include "parameters.h"

using Idx = Kokkos::RangePolicy<>::index_type;
using Index_t = Kokkos::IndexType<Idx>;
using Team_t = Kokkos::TeamPolicy<Index_t>::member_type;

KOKKOS_INLINE_FUNCTION int index_1D(const Params& p, int i, int j)
{
    return p.index_start + j * p.idx_row + i * p.idx_col;
}

std::tuple<int, int> real_domain(const Params& p);
std::tuple<int, int> real_domain_fluxes(const Params& p);
std::tuple<int, int> real_domain_advection(const Params& p);
std::tuple<int, int> zero_to(int i);
std::tuple<int, int> all_cells(const Params& p);


KOKKOS_INLINE_FUNCTION Kokkos::RangePolicy<Index_t> iter(const std::tuple<int, int>&& range)
{
    auto [deb, fin] = range;
    return {static_cast<Idx>(deb), static_cast<Idx>(fin+1)}; // +1 as RangePolicy is an open interval
}


KOKKOS_INLINE_FUNCTION Kokkos::RangePolicy<Index_t> iter(const std::tuple<int, int>& range)
{
    return iter(std::forward<const std::tuple<int, int>>(range));
}


KOKKOS_INLINE_FUNCTION Kokkos::TeamPolicy<Index_t> iter_simd(const std::tuple<int, int>&& range)
{
    auto [deb, fin] = range;
    int size = fin - deb + 1;
    return { size, Kokkos::AUTO };
}


KOKKOS_INLINE_FUNCTION Kokkos::TeamPolicy<Index_t> iter_simd(const std::tuple<int, int>& range)
{
    return iter_simd(std::forward<const std::tuple<int, int>>(range));
}


KOKKOS_INLINE_FUNCTION Idx iter_team_start(const Team_t& team, const std::tuple<int, int>& range)
{
    return team.league_rank() * team.team_size() + std::get<0>(range);
}


#endif //ARMON_KOKKOS_INDEXING_H
