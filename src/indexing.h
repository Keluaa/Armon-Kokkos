
#ifndef ARMON_KOKKOS_INDEXING_H
#define ARMON_KOKKOS_INDEXING_H

#include "Kokkos_Core.hpp"

#include "parameters.h"

using Idx = Kokkos::RangePolicy<>::index_type;
using Index_t = Kokkos::IndexType<Idx>;
using Team_t = Kokkos::TeamPolicy<Index_t>::member_type;


struct Range {
    Idx start;
    Idx end;  // exclusive (open interval)

    [[nodiscard]] Idx length() const { return end - start; }
};


KOKKOS_INLINE_FUNCTION int index_1D(const Params& p, int i, int j)
{
    return p.index_start + j * p.idx_row + i * p.idx_col;
}

Range real_domain(const Params& p);
Range real_domain_fluxes(const Params& p);
Range real_domain_advection(const Params& p);
Range zero_to(int i);
Range all_cells(const Params& p);


KOKKOS_INLINE_FUNCTION Kokkos::RangePolicy<Index_t> iter(const Range&& range)
{
    return {range.start, range.end };
}


KOKKOS_INLINE_FUNCTION Kokkos::RangePolicy<Index_t> iter(const Range& range)
{
    return iter(std::forward<const Range>(range));
}


KOKKOS_INLINE_FUNCTION Kokkos::TeamPolicy<Index_t> iter_simd(const Range&& range, int V)
{
    int size = static_cast<int>(Kokkos::ceil(static_cast<double>(range.length()) / V));
    return { size, Kokkos::AUTO, V };
}


KOKKOS_INLINE_FUNCTION Kokkos::TeamPolicy<Index_t> iter_simd(const Range& range, int V)
{
    return iter_simd(std::forward<const Range>(range), V);
}


KOKKOS_INLINE_FUNCTION Idx iter_team_start(const Team_t& team, const Range& range)
{
    return team.league_rank() * team.team_size() + range.start;
}


#endif //ARMON_KOKKOS_INDEXING_H
