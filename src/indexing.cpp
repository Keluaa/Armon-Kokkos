
#include "indexing.h"


std::tuple<int, int> real_domain(const Params& p)
{
    int deb = p.ideb;
    int fin = p.ifin;

    if (p.single_comm_per_axis_pass) {
        int r = p.extra_ring_width;

        // Add 'r' columns/rows on each side of the domain
        deb -= r * p.row_length + r;
        fin += r * p.row_length + r;
    }

    return std::make_tuple(deb, fin);
}


std::tuple<int, int> real_domain_fluxes(const Params& p)
{
    auto [deb, fin] = real_domain(p);

    // Add one row/column along the current direction
    fin += p.s;

    return std::make_tuple(deb, fin);
}


std::tuple<int, int> real_domain_advection(const Params& p)
{
    auto [deb, fin] = real_domain(p);

    // Add one row or column on each side of the current direction
    deb -= p.s;
    fin += p.s;

    return std::make_tuple(deb, fin);
}


std::tuple<int, int> zero_to(int i)
{
    return std::make_tuple(0, i);
}


std::tuple<int, int> all_cells(const Params& p)
{
    return zero_to(p.nb_cells - 1);
}


Kokkos::RangePolicy<Index_t> iter(std::tuple<int, int>&& range)
{
    auto [deb, fin] = range;
    return {deb, fin+1}; // +1 as RangePolicy is an open interval
}


Kokkos::RangePolicy<Index_t> iter(std::tuple<int, int>& range)
{
    return iter(std::forward<std::tuple<int, int>>(range));
}
