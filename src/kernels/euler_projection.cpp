
#include <Kokkos_Core.hpp>

#include "kernels.h"
#include "parallel_kernels.h"


extern "C"
void cell_update(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                const view& ustar, const view& pstar, const view& Emat,
                view& rho, view& u)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);
        flt_t dm = rho[i] * dx;
        rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]));
        u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             );
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]);
    });
}


extern "C"
void euler_projection(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                      const view& ustar, view& rho, view& umat, view& vmat, view& Emat,
                      const view& advection_rho, const view& advection_urho,
                      const view& advection_vrho, const view& advection_Erho)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);

        flt_t dX = dx + dt * (ustar[i+s] - ustar[i]);

        flt_t tmp_rho  = (dX * rho[i]           - (advection_rho[i+s]  - advection_rho[i] )) / dx;
        flt_t tmp_urho = (dX * rho[i] * umat[i] - (advection_urho[i+s] - advection_urho[i])) / dx;
        flt_t tmp_vrho = (dX * rho[i] * vmat[i] - (advection_vrho[i+s] - advection_vrho[i])) / dx;
        flt_t tmp_Erho = (dX * rho[i] * Emat[i] - (advection_Erho[i+s] - advection_Erho[i])) / dx;

        rho[i]  = tmp_rho;
        umat[i] = tmp_urho / tmp_rho;
        vmat[i] = tmp_vrho / tmp_rho;
        Emat[i] = tmp_Erho / tmp_rho;
    });
}


extern "C"
void advection_first_order(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dt,
                           const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
                           view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);

        flt_t disp = dt * ustar[i];
        Idx is = i;
        i -= Idx(disp > 0) * s;

        advection_rho[is]  = disp * (rho[i]          );
        advection_urho[is] = disp * (rho[i] * umat[i]);
        advection_vrho[is] = disp * (rho[i] * vmat[i]);
        advection_Erho[is] = disp * (rho[i] * Emat[i]);
    });
}


KOKKOS_INLINE_FUNCTION flt_t slope_minmod(flt_t u_im, flt_t u_i, flt_t u_ip, flt_t r_m, flt_t r_p)
{
    flt_t D_u_p = r_p * (u_ip - u_i );
    flt_t D_u_m = r_m * (u_i  - u_im);
    flt_t s = Kokkos::copysign(flt_t(1), D_u_p);
    return s * Kokkos::max(flt_t(0), Kokkos::min(s * D_u_p, s * D_u_m));
}


extern "C"
void advection_second_order(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                            const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
                            view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);

        Idx is = i;
        flt_t disp = dt * ustar[i];
        flt_t Dx;
        if (disp > 0) {
            Dx = -(dx - dt * ustar[i-s]);
            i = i - s;
        } else {
            Dx = dx + dt * ustar[i+s];
        }

        flt_t Dx_lm = dx + dt * (ustar[i]     - ustar[i-s]);
        flt_t Dx_l  = dx + dt * (ustar[i+s]   - ustar[i]  );
        flt_t Dx_lp = dx + dt * (ustar[i+2*s] - ustar[i+s]);

        flt_t r_m = (2 * Dx_l) / (Dx_l + Dx_lm);
        flt_t r_p = (2 * Dx_l) / (Dx_l + Dx_lp);

        flt_t slope_r  = slope_minmod(rho[i-s]            , rho[i]          , rho[i+s]            , r_m, r_p);
        flt_t slope_ur = slope_minmod(rho[i-s] * umat[i-s], rho[i] * umat[i], rho[i+s] * umat[i+s], r_m, r_p);
        flt_t slope_vr = slope_minmod(rho[i-s] * vmat[i-s], rho[i] * vmat[i], rho[i+s] * vmat[i+s], r_m, r_p);
        flt_t slope_Er = slope_minmod(rho[i-s] * Emat[i-s], rho[i] * Emat[i], rho[i+s] * Emat[i+s], r_m, r_p);

        flt_t length_factor = Dx / (2 * Dx_l);
        advection_rho[is]  = disp * (rho[i]           - slope_r  * length_factor);
        advection_urho[is] = disp * (rho[i] * umat[i] - slope_ur * length_factor);
        advection_vrho[is] = disp * (rho[i] * vmat[i] - slope_vr * length_factor);
        advection_Erho[is] = disp * (rho[i] * Emat[i] - slope_Er * length_factor);
    });
}
