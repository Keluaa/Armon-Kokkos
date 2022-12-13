
#ifndef ARMON_KOKKOS_INDEXING_H
#define ARMON_KOKKOS_INDEXING_H

#include "Kokkos_Core.hpp"

#include "parameters.h"

using Index_t = Kokkos::IndexType<int>;

KOKKOS_INLINE_FUNCTION int index_1D(const Params& p, int i, int j)
{
    return p.index_start + j * p.idx_row + i * p.idx_col;
}

std::tuple<int, int> real_domain(const Params& p);
std::tuple<int, int> real_domain_fluxes(const Params& p);
std::tuple<int, int> real_domain_advection(const Params& p);
std::tuple<int, int> zero_to(int i);
std::tuple<int, int> all_cells(const Params& p);

Kokkos::RangePolicy<Index_t> iter(std::tuple<int, int>&& range);
Kokkos::RangePolicy<Index_t> iter(std::tuple<int, int>& range);

#endif //ARMON_KOKKOS_INDEXING_H
