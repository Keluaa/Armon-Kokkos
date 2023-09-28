
#include "indexing.h"


Range real_domain(const Params& p)
{
    Idx deb = p.ideb;
    Idx fin = p.ifin;

    if (p.single_comm_per_axis_pass) {
        Idx r = p.extra_ring_width;

        // Add 'r' columns/rows on each side of the domain
        deb -= r * p.row_length + r;
        fin += r * p.row_length + r;
    }

    return { deb, fin };
}


Range real_domain_fluxes(const Params& p)
{
    auto [start, end] = real_domain(p);

    // Add one row/column along the current direction
    end += p.s;

    return { start, end };
}


Range real_domain_advection(const Params& p)
{
    auto [start, end] = real_domain(p);

    // Add one row or column on each side of the current direction
    start -= p.s;
    end += p.s;

    return { start, end };
}


Range zero_to(int i)
{
    return { 0, static_cast<Idx>(i) };
}


Range all_cells(const Params& p)
{
    return zero_to(p.nb_cells - 1);
}
