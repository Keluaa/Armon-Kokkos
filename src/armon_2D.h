
#ifndef ARMON_KOKKOS_ARMON_2D_H
#define ARMON_KOKKOS_ARMON_2D_H

#include <Kokkos_Core.hpp>

#include "parameters.h"
#include "indexing.h"
#include "data.h"


// Program time contribution tracking
extern std::map<std::string, double> time_contribution;
#define CAT(a, b) a##b
#define TIC_IMPL(line_nb) auto CAT(tic_, line_nb) = std::chrono::steady_clock::now()
#define TAC_IMPL(label, line_nb) \
    auto CAT(tac_, line_nb) = std::chrono::steady_clock::now(); \
    double CAT(expr_time_, line_nb) = std::chrono::duration<double>(CAT(tac_, line_nb) - CAT(tic_, line_nb)).count(); \
    time_contribution[label]   += CAT(expr_time_, line_nb); \
    time_contribution["TOTAL"] += CAT(expr_time_, line_nb)
#define TIC() TIC_IMPL(__LINE__)
#define TAC(label) TAC_IMPL(label, __LINE__)


template <Limiter L>
KOKKOS_INLINE_FUNCTION flt_t limiter(flt_t)
{
    static_assert(L == Limiter::None, "Wrong limiter type");
    return flt_t(1);
}


template<>
KOKKOS_INLINE_FUNCTION flt_t limiter<Limiter::Minmod>(flt_t r)
{
    return std::max(flt_t(0), std::min(flt_t(1), r));
}


template<>
KOKKOS_INLINE_FUNCTION flt_t limiter<Limiter::Superbee>(flt_t r)
{
    return std::max(std::max(flt_t(0), std::min(flt_t(1), 2*r)), std::min(flt_t(2), r));
}


void numericalFluxes(const Params& p, Data& d, flt_t dt);

void perfectGasEOS(const Params& p, Data& d, flt_t gamma);
void bizarriumEOS(const Params& p, Data& d);
void update_EOS(const Params& p, Data& d);

void cellUpdate(const Params& p, Data& d, flt_t dt);

void init_test(Params& p, Data& d);

void boundaryConditions(const Params& p, Data& d, Side side);
void boundaryConditions(const Params& p, Data& d);

void euler_projection(const Params& p, Data& d, flt_t dt,
                      view& advection_rho, view& advection_urho,
                      view& advection_vrho, view& advection_Erho);
void advection_first_order(const Params& p, Data& d, flt_t dt,
                           view& advection_rho, view& advection_urho,
                           view& advection_vrho, view& advection_Erho);
void advection_second_order(const Params& p, Data& d, flt_t dt,
                            view& advection_rho, view& advection_urho,
                            view& advection_vrho, view& advection_Erho);
void projection_remap(const Params& p, Data& d, flt_t dt);

flt_t dtCFL(const Params& p, Data& d, flt_t dta);

void write_output(const Params& p, const HostData& d);

std::tuple<double, flt_t, int> time_loop(Params& p, Data& d);

bool armon(Params& params);


#endif //ARMON_KOKKOS_ARMON_2D_H
