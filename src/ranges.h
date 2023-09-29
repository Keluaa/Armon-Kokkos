
#ifndef ARMON_KOKKOS_RANGES_H
#define ARMON_KOKKOS_RANGES_H

#include "kernels/indexing.h"
#include "parameters.h"


/**
 * `DomainRange` over all real cells
 */
DomainRange real_domain(const Params& p);

/**
 * `DomainRange` over all cells which need to have their fluxes calculated
 */
DomainRange domain_fluxes(const Params& p);

/**
 * `DomainRange` over all cells which need to move in the lagrangian phase
 */
DomainRange domain_cell_update(const Params& p);

/**
 * `DomainRange` over all cells which need to have their advection fluxes calculated in the remap phase
 */
DomainRange domain_advection(const Params& p);

/**
 *  `DomainRange` over all cells, including ghosts
 */
DomainRange complete_domain(const Params& p);

/**
 *  `DomainRange` over all cells in the boundary region of `side`. `stride` and `disp` are iteration variables needed
 *  by the boundary conditions kernel.
 */
DomainRange boundary_conditions_domain(const Params& p, Side side, int& stride, int& disp);


/**
 * Transforms a 2D index `(i, j)` into a 1D array index.
 */
KOKKOS_INLINE_FUNCTION int index_1D(const Params& p, int i, int j)
{
    return p.index_start + j * p.idx_row + i * p.idx_col;
}

#endif //ARMON_KOKKOS_RANGES_H
