
#ifndef ARMON_KOKKOS_ARMON_2D_H
#define ARMON_KOKKOS_ARMON_2D_H

#include <Kokkos_Core.hpp>

#include "kernels/common.h"
#include "parameters.h"
#include "data.h"


void numerical_fluxes(const Params& p, Data& d, flt_t dt);

void update_EOS(const Params& p, Data& d);

void cell_update(const Params& p, Data& d, flt_t dt);

void init_test(const Params& p, Data& d, bool debug_indexes = false);

void boundary_conditions(const Params& p, Data& d, Side side);
void boundary_conditions(const Params& p, Data& d);

void euler_projection(const Params& p, Data& d, flt_t dt,
                      const view& advection_rho, const view& advection_urho,
                      const view& advection_vrho, const view& advection_Erho);
void advection_first_order(const Params& p, Data& d, flt_t dt,
                           view& advection_rho, view& advection_urho,
                           view& advection_vrho, view& advection_Erho);
void advection_second_order(const Params& p, Data& d, flt_t dt,
                            view& advection_rho, view& advection_urho,
                            view& advection_vrho, view& advection_Erho);
void projection_remap(const Params& p, Data& d, flt_t dt);

flt_t dt_CFL(const Params& p, Data& d, flt_t dta);

std::tuple<flt_t, flt_t> conservation_vars(const Params& p, Data& d);

std::tuple<double, flt_t, int> time_loop(Params& p, Data& d, HostData& hd);

bool armon(Params& params);


#endif //ARMON_KOKKOS_ARMON_2D_H
