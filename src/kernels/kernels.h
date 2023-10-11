
#ifndef ARMON_KOKKOS_KERNELS_H
#define ARMON_KOKKOS_KERNELS_H

#include "indexing.h"
#include "test_cases.h"
#include "limiters.h"


extern "C" DLL_EXPORT
void acoustic(const Range& range, const InnerRange2D& inner_range, Idx s,
              view& ustar, view& pstar,
              const view& rho, const view& u, const view& pmat, const view& cmat);

extern "C" DLL_EXPORT
void acoustic_GAD(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dt, flt_t dx,
                  view& ustar, view& pstar,
                  const view& rho, const view& u, const view& pmat, const view& cmat,
                  Limiter limiter);

extern "C" DLL_EXPORT
void perfect_gas_EOS(const Range& range, const InnerRange2D& inner_range, flt_t gamma,
                     const view& rho, const view& Emat, const view& umat, const view& vmat,
                     view& pmat, view& cmat, view& gmat);

extern "C" DLL_EXPORT
void bizarrium_EOS(const Range& range, const InnerRange2D& inner_range,
                   const view& rho, const view& umat, const view& vmat, const view& Emat,
                   view& pmat, view& cmat, view& gmat);

extern "C" DLL_EXPORT
void cell_update(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                 const view& ustar, const view& pstar, view& rho, view& u, view& Emat);

extern "C" DLL_EXPORT
void euler_projection(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                      const view& ustar, view& rho, view& umat, view& vmat, view& Emat,
                      const view& advection_rho, const view& advection_urho,
                      const view& advection_vrho, const view& advection_Erho);

extern "C" DLL_EXPORT
void advection_first_order(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dt,
                           const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
                           view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho);

extern "C" DLL_EXPORT
void advection_second_order(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                            const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
                            view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho);

extern "C" DLL_EXPORT
void boundary_conditions(const Range& range, const InnerRange1D& inner_range,
                         Idx stencil_width, Idx disp,
                         flt_t u_factor, flt_t v_factor,
                         view& rho, view& umat, view& vmat, view& pmat, view& cmat, view& gmat, view& Emat);

extern "C" DLL_EXPORT
void read_border_array(const Range& range, const InnerRange2D& inner_range,
                       Idx side_length, Idx nghost,
                       const view& rho, const view& umat, const view& vmat, const view& pmat,
                       const view& cmat, const view& gmat, const view& Emat,
                       view& value_array);

extern "C" DLL_EXPORT
void write_border_array(const Range& range, const InnerRange2D& inner_range,
                        Idx side_length, Idx nghost,
                        view& rho, view& umat, view& vmat, view& pmat,
                        view& cmat, view& gmat, view& Emat,
                        const view& value_array);

extern "C" DLL_EXPORT
void init_test(const Range& range, const InnerRange1D& inner_range,
               Idx row_length, Idx nb_ghosts, Idx nx, Idx ny, flt_t sx,
               flt_t sy, flt_t ox, flt_t oy, Idx pos_x, Idx pos_y, Idx g_nx, Idx g_ny,
               view& x, view& y, view& rho, view& Emat, view& umat, view& vmat, mask_view& domain_mask,
               view& pmat, view& cmat, view& ustar, view& pstar,
               Test test, bool debug_indexes, flt_t test_option);

extern "C" DLL_EXPORT
flt_t dt_CFL(const Range& range, const InnerRange1D& inner_range, flt_t dx, flt_t dy,
             const view& umat, const view& vmat, const view& cmat, const mask_view& domain_mask);

extern "C" DLL_EXPORT
void conservation_vars(const Range& range, const InnerRange1D& inner_range, flt_t dx,
                       const view& rho, const view& Emat, const mask_view& domain_mask,
                       flt_t& total_mass, flt_t& total_energy);

#endif //ARMON_KOKKOS_KERNELS_H
