
#include "ranges.h"


DomainRange real_domain(const Params& p)
{
    return {
        index_1D(p, 0, 0), p.row_length, index_1D(p, 0, p.ny - 1),
        0, 1, p.nx - 1
    };
}


DomainRange domain_fluxes(const Params& p)
{
    DomainRange dr = real_domain(p);

    long extra = 1;
    if (p.projection == Projection::Euler_2nd) {
        extra += 1;
    }

    // Extra cells because of the stencil
    dr.inflate_dir(p.current_axis, extra);

    // Fluxes are computed between 'i-s' and 'i', we need one more cell on the right to have all fluxes
    dr.expand_dir(p.current_axis, 1);

    return dr;
}


DomainRange domain_cell_update(const Params& p)
{
    DomainRange dr = real_domain(p);

    long extra = 1;
    if (p.projection == Projection::Euler_2nd) {
        extra += 1;
    }

    // Extra cells because of the stencil
    dr.inflate_dir(p.current_axis, extra);

    return dr;
}


DomainRange domain_advection(const Params& p)
{
    DomainRange dr = real_domain(p);
    dr.expand_dir(p.current_axis, 1);
    return dr;
}


DomainRange complete_domain(const Params& p)
{
    DomainRange dr = real_domain(p);
    dr.inflate_dir(Axis::X, p.nb_ghosts);
    dr.inflate_dir(Axis::Y, p.nb_ghosts);
    return dr;
}


DomainRange boundary_conditions_domain(const Params& p, Side side, int& disp)
{
    int i_start, stride, loop_range;

    switch (side) {
    case Side::Left:
        stride = p.row_length;
        i_start = index_1D(p, -1, 0);
        loop_range = p.ny;
        disp = 1;
        break;

    case Side::Right:
        stride = p.row_length;
        i_start = index_1D(p, p.nx, 0);
        loop_range = p.ny;
        disp = -1;
        break;

    case Side::Top:
        stride = 1;
        i_start = index_1D(p, 0, p.ny);
        loop_range = p.nx;
        disp = -p.row_length;
        break;

    case Side::Bottom:
        stride = 1;
        i_start = index_1D(p, 0, -1);
        loop_range = p.nx;
        disp = p.row_length;
        break;
    }

    // length() == loop_range
    // begin()  == i_start
    return {
        0, 1, 0,
        i_start, stride, i_start + loop_range * stride - stride
    };
}
