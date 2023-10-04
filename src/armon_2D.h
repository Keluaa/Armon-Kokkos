
#ifndef ARMON_KOKKOS_ARMON_2D_H
#define ARMON_KOKKOS_ARMON_2D_H

#include <Kokkos_Core.hpp>

#include "kernels/common.h"
#include "parameters.h"
#include "data.h"


void numerical_fluxes(const Params& p, Data& d);

void update_EOS(const Params& p, Data& d);

void cell_update(const Params& p, Data& d);

void init_test(const Params& p, Data& d, bool debug_indexes = false);

void boundary_conditions(const Params& p, Data& d, Side side);
void boundary_conditions(const Params& p, Data& d);

void euler_projection(const Params& p, Data& d,
                      const view& advection_rho, const view& advection_urho,
                      const view& advection_vrho, const view& advection_Erho);
void advection_first_order(const Params& p, Data& d,
                           view& advection_rho, view& advection_urho,
                           view& advection_vrho, view& advection_Erho);
void advection_second_order(const Params& p, Data& d,
                            view& advection_rho, view& advection_urho,
                            view& advection_vrho, view& advection_Erho);
void projection_remap(const Params& p, Data& d);

flt_t local_time_step(const Params& p, Data& d, flt_t prev_dt);
void time_step(Params& p, Data& d);

std::tuple<flt_t, flt_t> conservation_vars(const Params& p, Data& d);

std::tuple<double, flt_t, int> time_loop(Params& p, Data& d, HostData& hd);

bool armon(Params& params);


#endif //ARMON_KOKKOS_ARMON_2D_H
