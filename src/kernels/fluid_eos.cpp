
#include <Kokkos_Core.hpp>

#include "kernels.h"
#include "parallel_kernels.h"


extern "C"
void perfect_gas_EOS(const Range& range, const InnerRange2D& inner_range, flt_t gamma,
                     const view& rho, const view& umat, const view& vmat, const view& Emat,
                     view& cmat, view& pmat, view& gmat)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);
        flt_t e = Emat[i] - flt_t(0.5) * (Kokkos::pow(umat[i], flt_t(2)) + Kokkos::pow(vmat[i], flt_t(2)));
        pmat[i] = (gamma - 1) * rho[i] * e;
        cmat[i] = Kokkos::sqrt(gamma * pmat[i] / rho[i]);
        gmat[i] = (1 + gamma) / 2;
    });
}


extern "C"
void bizarrium_EOS(const Range& range, const InnerRange2D& inner_range,
                   const view& rho, const view& umat, const view& vmat, const view& Emat,
                   view& cmat, view& pmat, view& gmat)
{
    const flt_t rho0 = 1e4, K0 = 1e11, Cv0 = 1e3, T0 = 300, eps0 = 0;
    const flt_t G0 = 1.5, s = 1.5, q = -42080895./14941154., r = 727668333./149411540.;

    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);

        flt_t x = rho[i] / rho0 - 1;
        flt_t g = G0 * (1 - rho0 / rho[i]);

        flt_t f0 = (1+(s/3-2)*x+q*(x*x)+r*(x*x*x))/(1-s*x);
        flt_t f1 = (s/3-2+2*q*x+3*r*(x*x)+s*f0)/(1-s*x);
        flt_t f2 = (2*q+6*r*x+2*s*f1)/(1-s*x);
        flt_t f3 = (6*r+3*s*f2)/(1-s*x);

        flt_t eps_k0 = eps0 - Cv0*T0*(1+g) + flt_t(0.5)*(K0/rho0)*(x*x)*f0;
        flt_t pk0 = -Cv0*T0*G0*rho0 + (flt_t(0.5)*K0*x*(1+x)*(1+x)*(2*f0+x*f1));
        flt_t pk0prime = -flt_t(0.5) * K0 * Kokkos::pow(1+x,flt_t(3))
                * rho0 * (2 * (1+3*x) * f0 + 2*x*(2+3*x) * f1 + (x*x) * (1+x) * f2);
        flt_t pk0second = flt_t(0.5) * K0 * Kokkos::pow(1+x,flt_t(4)) * (rho0*rho0)
                * (12*(1+2*x)*f0 + 6*(1+6*x+6*(x*x)) * f1 + 6*x*(1+x)*(1+2*x) * f2
                   + Kokkos::pow(x*(1+x),flt_t(2)) * f3);

        flt_t e = Emat[i] - flt_t(0.5) * (Kokkos::pow(umat[i], flt_t(2)) + Kokkos::pow(vmat[i], flt_t(2)));
        pmat[i] = pk0 + G0*rho0*(e - eps_k0);
        cmat[i] = Kokkos::sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i];
        gmat[i] = flt_t(0.5) / (Kokkos::pow(rho[i],flt_t(3)) * Kokkos::pow(cmat[i],flt_t(2)))
                * (pk0second + Kokkos::pow(G0 * rho0,flt_t(2)) * (pmat[i]-pk0));
    });
}
