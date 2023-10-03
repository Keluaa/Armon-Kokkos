
#ifndef ARMON_KOKKOS_KERNELS_H
#define ARMON_KOKKOS_KERNELS_H

#include "indexing.h"
#include "test_cases.h"
#include "limiters.h"


extern "C"
void acoustic(const Range& range, const InnerRange2D& inner_range, Idx s,
              const view& rho, const view& u, const view& cmat, const view& pmat,
              view& ustar, view& pstar);

extern "C"
void acoustic_GAD(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                  const view& rho, const view& u, const view& cmat, const view& pmat,
                  view& ustar, view& pstar,
                  Limiter limiter);

extern "C"
void perfect_gas_EOS(const Range& range, const InnerRange2D& inner_range, flt_t gamma,
                     const view& rho, const view& umat, const view& vmat, const view& Emat,
                     view& cmat, view& pmat, view& gmat);

extern "C"
void bizarrium_EOS(const Range& range, const InnerRange2D& inner_range,
                   const view& rho, const view& umat, const view& vmat, const view& Emat,
                   view& cmat, view& pmat, view& gmat);

extern "C"
void cell_update(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                 const view& ustar, const view& pstar, const view& Emat,
                 view& rho, view& u);

extern "C"
void euler_projection(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                      const view& ustar, view& rho, view& umat, view& vmat, view& Emat,
                      const view& advection_rho, const view& advection_urho,
                      const view& advection_vrho, const view& advection_Erho);

extern "C"
void advection_first_order(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dt,
                           const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
                           view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho);

extern "C"
void advection_second_order(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                            const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
                            view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho);

extern "C"
void boundary_conditions(const Range& range, const InnerRange1D& inner_range,
                         Idx disp, Idx stencil_width,
                         flt_t u_factor, flt_t v_factor,
                         view& rho, view& umat, view& vmat, view& pmat, view& cmat, view& gmat, view& Emat);

extern "C"
void read_border_array(const Range& range, const InnerRange1D& inner_range,
                       Idx nghost, Idx side_length,
                       const view& rho, const view& umat, const view& vmat, const view& pmat,
                       const view& cmat, const view& gmat, const view& Emat,
                       view& value_array);

extern "C"
void write_border_array(const Range& range, const InnerRange1D& inner_range,
                        Idx nghost, Idx side_length,
                        view& rho, view& umat, view& vmat, view& pmat,
                        view& cmat, view& gmat, view& Emat,
                        const view& value_array);

extern "C"
void init_test(const Range& range, const InnerRange1D& inner_range,
               Idx row_length, Idx nb_ghosts,
               Idx nx, Idx ny, Idx g_nx, Idx g_ny, Idx pos_x, Idx pos_y,
               flt_t sx, flt_t sy, flt_t ox, flt_t oy,
               view& x, view& y, view& rho, view& Emat, view& umat, view& vmat,
               mask_view& domain_mask, view& pmat, view& cmat, view& ustar, view& pstar,
               Test test, bool debug_indexes, flt_t test_option);

extern "C"
flt_t dt_CFL(const Range& range, const InnerRange1D& inner_range, flt_t dx, flt_t dy,
             const view& umat, const view& vmat, const view& cmat, const mask_view& domain_mask);

extern "C"
void conservation_vars(const Range& range, const InnerRange1D& inner_range, flt_t dx,
                       const view& rho, const view& Emat, const mask_view& domain_mask,
                       flt_t& total_mass, flt_t& total_energy);

#endif //ARMON_KOKKOS_KERNELS_H
