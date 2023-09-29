
#include <Kokkos_Core.hpp>

#include "kernels.h"
#include "parallel_kernels.h"
#include "limiters.h"


KOKKOS_INLINE_FUNCTION std::tuple<flt_t, flt_t> acoustic_Godunov(
        flt_t rho_i, flt_t rho_im, flt_t c_i, flt_t c_im,
        flt_t u_i,   flt_t u_im,   flt_t p_i, flt_t p_im)
{
    flt_t rc_l = rho_im * c_im;
    flt_t rc_r = rho_i  * c_i;
    flt_t ustar_i = (rc_l * u_im + rc_r * u_i +               (p_im - p_i)) / (rc_l + rc_r);
    flt_t pstar_i = (rc_r * p_im + rc_l * p_i + rc_l * rc_r * (u_im - u_i)) / (rc_l + rc_r);
    return std::make_tuple(ustar_i, pstar_i);
}


extern "C"
void acoustic(const Range& range, const InnerRange2D& inner_range, Idx s,
              const view& rho, const view& u, const view& cmat, const view& pmat,
              view& ustar, view& pstar)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);
        auto [ustar_i, pstar_i] = acoustic_Godunov(
            rho[i], rho[i-s], cmat[i], cmat[i-s],
              u[i],   u[i-s], pmat[i], pmat[i-s]);
        ustar[i] = ustar_i;
        pstar[i] = pstar_i;
    });
}


template<Limiter L>
void acoustic_GAD(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                  const view& rho, const view& u, const view& cmat, const view& pmat,
                  view& ustar, view& pstar)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);

        // First order acoustic solver on the left cell
        auto [ustar_im, pstar_im] = acoustic_Godunov(
            rho[i-s], rho[i-2*s], cmat[i-s], cmat[i-2*s],
              u[i-s],   u[i-2*s], pmat[i-s], pmat[i-2*s]);

        // First order acoustic solver on the current cell
        auto [ustar_i, pstar_i] = acoustic_Godunov(
            rho[i], rho[i-s], cmat[i], cmat[i-s],
              u[i],   u[i-s], pmat[i], pmat[i-s]);

        // First order acoustic solver on the right cell
        auto [ustar_ip, pstar_ip] = acoustic_Godunov(
            rho[i+s], rho[i], cmat[i+s], cmat[i],
              u[i+s],   u[i], pmat[i+s], pmat[i]);

        // Second order GAD acoustic solver on the current cell

        flt_t r_um = (ustar_ip -      u[i]) / (ustar_i -    u[i-s] + flt_t(1e-6));
        flt_t r_pm = (pstar_ip -   pmat[i]) / (pstar_i - pmat[i-s] + flt_t(1e-6));
        flt_t r_up = (   u[i-s] - ustar_im) / (   u[i] -   ustar_i + flt_t(1e-6));
        flt_t r_pp = (pmat[i-s] - pstar_im) / (pmat[i] -   pstar_i + flt_t(1e-6));

        r_um = limiter<L>(r_um);
        r_pm = limiter<L>(r_pm);
        r_up = limiter<L>(r_up);
        r_pp = limiter<L>(r_pp);

        flt_t dm_l = rho[i-s] * dx;
        flt_t dm_r = rho[i]   * dx;
        flt_t Dm   = (dm_l + dm_r) / 2;

        flt_t rc_l  = rho[i-s] * cmat[i-s];
        flt_t rc_r  = rho[i]   * cmat[i];
        flt_t theta = flt_t(0.5) * (1 - (rc_l + rc_r) / 2 * (dt / Dm));

        ustar[i] = ustar_i + theta * (r_up * (   u[i] - ustar_i) - r_um * (ustar_i -    u[i-s]));
        pstar[i] = pstar_i + theta * (r_pp * (pmat[i] - pstar_i) - r_pm * (pstar_i - pmat[i-s]));
    });
}


extern "C"
void acoustic_GAD(const Range& range, const InnerRange2D& inner_range, Idx s, flt_t dx, flt_t dt,
                  const view& rho, const view& u, const view& cmat, const view& pmat,
                  view& ustar, view& pstar,
                  Limiter limiter)
{
    switch (limiter) {
        case Limiter::None:     return acoustic_GAD<Limiter::None    >(range, inner_range, s, dx, dt, rho, u, cmat, pmat, ustar, pstar);
        case Limiter::Minmod:   return acoustic_GAD<Limiter::Minmod  >(range, inner_range, s, dx, dt, rho, u, cmat, pmat, ustar, pstar);
        case Limiter::Superbee: return acoustic_GAD<Limiter::Superbee>(range, inner_range, s, dx, dt, rho, u, cmat, pmat, ustar, pstar);
        default:
            throw std::out_of_range(
                    "Invalid limiter index: " + std::to_string(static_cast<int>(limiter))
                    + ", expected a value between " + std::to_string(static_cast<int>(Limiter::None))
                    + " and " + std::to_string(static_cast<int>(Limiter::Superbee))
            );
    }
}
