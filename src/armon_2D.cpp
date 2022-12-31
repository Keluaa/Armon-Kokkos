
#include "armon_2D.h"

#include "io.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <string>
#include <array>


std::map<std::string, double> time_contribution;


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


void acoustic(const Params& p, Data& d, const view& u)
{
    const int s = p.s;
    Kokkos::parallel_for(iter(real_domain_fluxes(p)),
    KOKKOS_LAMBDA(const int i) {
        auto [ustar_i, pstar_i] = acoustic_Godunov(
            d.rho[i], d.rho[i-s], d.cmat[i], d.cmat[i-s],
                u[i],     u[i-s], d.pmat[i], d.pmat[i-s]);
        d.ustar[i] = ustar_i;
        d.pstar[i] = pstar_i;
    });
}


template<Limiter L>
void acoustic_GAD(const Params& p, Data& d, flt_t dt, const view& u)
{
    const int s = p.s;
    const flt_t dx = p.dx;

    Kokkos::parallel_for(iter(real_domain_fluxes(p)),
    KOKKOS_LAMBDA(const int i) {
        // First order acoustic solver on the left cell
        auto [ustar_im, pstar_im] = acoustic_Godunov(
                d.rho[i-s], d.rho[i-2*s], d.cmat[i-s], d.cmat[i-2*s],
                    u[i-s],     u[i-2*s], d.pmat[i-s], d.pmat[i-2*s]);

        // First order acoustic solver on the current cell
        auto [ustar_i, pstar_i] = acoustic_Godunov(
                d.rho[i], d.rho[i-s], d.cmat[i], d.cmat[i-s],
                    u[i],     u[i-s], d.pmat[i], d.pmat[i-s]);

        // First order acoustic solver on the right cell
        auto [ustar_ip, pstar_ip] = acoustic_Godunov(
                d.rho[i+s], d.rho[i], d.cmat[i+s], d.cmat[i],
                    u[i+s],     u[i], d.pmat[i+s], d.pmat[i]);

        // Second order GAD acoustic solver on the current cell

        flt_t r_um = (ustar_ip -        u[i]) / (ustar_i -      u[i-s] + flt_t(1e-6));
        flt_t r_pm = (pstar_ip -   d.pmat[i]) / (pstar_i - d.pmat[i-s] + flt_t(1e-6));
        flt_t r_up = (     u[i-s] - ustar_im) / (     u[i] -   ustar_i + flt_t(1e-6));
        flt_t r_pp = (d.pmat[i-s] - pstar_im) / (d.pmat[i] -   pstar_i + flt_t(1e-6));

        r_um = limiter<L>(r_um);
        r_pm = limiter<L>(r_pm);
        r_up = limiter<L>(r_up);
        r_pp = limiter<L>(r_pp);

        flt_t dm_l = d.rho[i-s] * dx;
        flt_t dm_r = d.rho[i]   * dx;
        flt_t Dm   = (dm_l + dm_r) / 2;

        flt_t rc_l  = d.rho[i-s] * d.cmat[i-s];
        flt_t rc_r  = d.rho[i]   * d.cmat[i];
        flt_t theta = flt_t(0.5) * (1 - (rc_l + rc_r) / 2 * (dt / Dm));

        d.ustar[i] = ustar_i + theta * (r_up * (     u[i] - ustar_i) - r_um * (ustar_i -      u[i-s]));
        d.pstar[i] = pstar_i + theta * (r_pp * (d.pmat[i] - pstar_i) - r_pm * (pstar_i - d.pmat[i-s]));
    });
}


void numericalFluxes(const Params& p, Data& d, flt_t dt)
{
    const view& u = p.current_axis == Axis::X ? d.umat : d.vmat;

    switch (p.riemann) {
    case Riemann::Acoustic:
        switch (p.scheme) {
        case Scheme::Godunov:       acoustic(p, d, u); break;
        case Scheme::GAD:
            switch (p.limiter) {
            case Limiter::None:     acoustic_GAD<Limiter::None    >(p, d, dt, u); break;
            case Limiter::Minmod:   acoustic_GAD<Limiter::Minmod  >(p, d, dt, u); break;
            case Limiter::Superbee: acoustic_GAD<Limiter::Superbee>(p, d, dt, u); break;
            }
            break;
        }
    }
}


void perfectGasEOS(const Params& p, Data& d, flt_t gamma)
{
    Kokkos::parallel_for(iter(real_domain(p)),
    KOKKOS_LAMBDA(const int i) {
        flt_t e = d.Emat[i] - flt_t(0.5) * (std::pow(d.umat[i], flt_t(2)) + std::pow(d.vmat[i], flt_t(2)));
        d.pmat[i] = (gamma - 1) * d.rho[i] * e;
        d.cmat[i] = std::sqrt(gamma * d.pmat[i] / d.rho[i]);
        d.gmat[i] = (1 + gamma) / 2;
    });
}


void bizarriumEOS(const Params& p, Data& d)
{
    const flt_t rho0 = 1e4, K0 = 1e11, Cv0 = 1e3, T0 = 300, eps0 = 0;
    const flt_t G0 = 1.5, s = 1.5, q = -42080895./14941154., r = 727668333./149411540.;

    Kokkos::parallel_for(iter(real_domain(p)),
    KOKKOS_LAMBDA(const int i){
        flt_t x = d.rho[i] / rho0 - 1;
        flt_t g = G0 * (1 - rho0 / d.rho[i]);

        flt_t f0 = (1+(s/3-2)*x+q*(x*x)+r*(x*x*x))/(1-s*x);
        flt_t f1 = (s/3-2+2*q*x+3*r*(x*x)+s*f0)/(1-s*x);
        flt_t f2 = (2*q+6*r*x+2*s*f1)/(1-s*x);
        flt_t f3 = (6*r+3*s*f2)/(1-s*x);

        flt_t eps_k0 = eps0 - Cv0*T0*(1+g) + flt_t(0.5)*(K0/rho0)*(x*x)*f0;
        flt_t pk0 = -Cv0*T0*G0*rho0 + (flt_t(0.5)*K0*x*(1+x)*(1+x)*(2*f0+x*f1));
        flt_t pk0prime = -flt_t(0.5) * K0 * std::pow(1+x,flt_t(3))
                * rho0 * (2 * (1+3*x) * f0 + 2*x*(2+3*x) * f1 + (x*x) * (1+x) * f2);
        flt_t pk0second = flt_t(0.5) * K0 * std::pow(1+x,flt_t(4)) * (rho0*rho0)
                * (12*(1+2*x)*f0 + 6*(1+6*x+6*(x*x)) * f1 + 6*x*(1+x)*(1+2*x) * f2
                   + std::pow(x*(1+x),flt_t(2)) * f3);

        flt_t e = d.Emat[i] - flt_t(0.5) * (std::pow(d.umat[i], flt_t(2)) + std::pow(d.vmat[i], flt_t(2)));
        d.pmat[i] = pk0 + G0*rho0*(e - eps_k0);
        d.cmat[i] = std::sqrt(G0*rho0*(d.pmat[i] - pk0) - pk0prime) / d.rho[i];
        d.gmat[i] = flt_t(0.5) / (std::pow(d.rho[i],flt_t(3)) * std::pow(d.cmat[i],flt_t(2)))
                * (pk0second + std::pow(G0 * rho0,flt_t(2)) * (d.pmat[i]-pk0));
    });
}


void update_EOS(const Params& p, Data& d)
{
    switch (p.test) {
    case Test::Sod:
    case Test::Sod_y:
    case Test::Sod_circ:
    {
        const flt_t gamma = 1.4;
        perfectGasEOS(p, d, gamma);
        break;
    }
    case Test::Bizarrium:
        bizarriumEOS(p, d);
        break;
    }
}


void cellUpdate(const Params& p, Data& d, flt_t dt)
{
    const int s = p.s;
    const flt_t dx = p.dx;
    view& u = p.current_axis == Axis::X ? d.umat : d.vmat;
    view& x = p.current_axis == Axis::X ? d.x : d.y;

    Kokkos::parallel_for(iter(real_domain(p)),
    KOKKOS_LAMBDA(const int i) {
        flt_t mask = d.domain_mask[i] * d.domain_mask[i+s] * d.domain_mask[i-s];
        flt_t dm = d.rho[i] * dx;
        d.rho[i]   = dm / (dx + dt * (d.ustar[i+s] - d.ustar[i]) * mask);
        u[i]      += dt / dm * (d.pstar[i]              - d.pstar[i+s]               ) * mask;
        d.Emat[i] += dt / dm * (d.pstar[i] * d.ustar[i] - d.pstar[i+s] * d.ustar[i+s]) * mask;
    });

    if (p.projection == Projection::None) {
        Kokkos::parallel_for(iter(real_domain(p)),
        KOKKOS_LAMBDA(const int i) {
            x[i] += dt * d.ustar[i];
        });
    }
}


void init_test(const Params& p, Data& d)
{
    const TestInitParams tp = get_test_init_params(p.test);

    flt_t sx = p.domain_size[0];
    flt_t sy = p.domain_size[1];

    flt_t ox = p.domain_origin[0];
    flt_t oy = p.domain_origin[1];

    bool one_more_ring = p.single_comm_per_axis_pass;
    int r = p.extra_ring_width;

    Kokkos::parallel_for(iter(all_cells(p)),
    KOKKOS_LAMBDA(const int i) {
        int ix = (i % p.row_length) - p.nb_ghosts;
        int iy = (i / p.row_length) - p.nb_ghosts;

        d.x[i] = flt_t(ix) / flt_t(p.nx) * sx + ox;
        d.y[i] = flt_t(iy) / flt_t(p.ny) * sy + oy;

        flt_t x_mid = d.x[i] + sx / (2 * p.nx);
        flt_t y_mid = d.y[i] + sy / (2 * p.ny);

        if (test_region_high(p.test, x_mid, y_mid)) {
            d.rho[i] = tp.high_rho;
            d.Emat[i] = tp.high_p / ((tp.gamma - 1) * tp.high_rho);
            d.umat[i] = tp.high_u;
            d.vmat[i] = tp.high_v;
        }
        else {
            d.rho[i] = tp.low_rho;
            d.Emat[i] = tp.low_p / ((tp.gamma - 1) * tp.low_rho);
            d.umat[i] = tp.low_u;
            d.vmat[i] = tp.low_v;
        }

        if (one_more_ring) {
            d.domain_mask[i] = (
                   ((-r <= ix) && (ix < p.nx+r) && (-r <= iy) && (iy < p.ny+r))  // Include as well a ring of ghost cells...
                && (( 0 <= ix) && (ix < p.nx)   || (0  <= iy) && (iy < p.ny)  )  // ...while excluding the corners of the subdomain
            );
        }
        else {
            d.domain_mask[i] = (0 <= ix && ix < p.nx && 0 <= iy && iy < p.ny);
        }

        // Set to zero to make sure no non-initialized values changes the result
        d.pmat[i] = 0;
        d.cmat[i] = 1;  // Set to 1 as a max speed of 0 will create NaNs
        d.ustar[i] = 0;
        d.pstar[i] = 0;
    });
}


void boundaryConditions(const Params& p, Data& d, Side side)
{
    flt_t u_factor = 1., v_factor = 1.;
    int stride = 1, disp = 1, i_start, loop_range;

    switch (side) {
    case Side::Left:
        if (p.test == Test::Sod || p.test == Test::Sod_circ) {
            u_factor = -1.;
        }
        stride = p.row_length;
        i_start = index_1D(p, -1, 0);
        loop_range = p.ny;
        disp = 1;
        break;

    case Side::Right:
        if (p.test == Test::Sod || p.test == Test::Sod_circ) {
            u_factor = -1.;
        }
        stride = p.row_length;
        i_start = index_1D(p, p.nx, 0);
        loop_range = p.ny;
        disp = -1;
        break;

    case Side::Top:
        if (p.test == Test::Sod_y || p.test == Test::Sod_circ) {
            v_factor = -1.;
        }
        stride = 1;
        i_start = index_1D(p, 0, p.ny);
        loop_range = p.nx;
        disp = -p.row_length;
        break;

    case Side::Bottom:
        if (p.test == Test::Sod_y || p.test == Test::Sod_circ) {
            v_factor = -1.;
        }
        stride = 1;
        i_start = index_1D(p, 0, -1);
        loop_range = p.nx;
        disp = p.row_length;
        break;
    }

    Kokkos::parallel_for(iter(zero_to(loop_range - 1)),
    KOKKOS_LAMBDA(const int idx) {
        int i = idx * stride + i_start;
        int ip = i + disp;

        for (int w = 0; w < p.stencil_width; w++) {
            d.rho[i]  = d.rho[ip];
            d.umat[i] = d.umat[ip] * u_factor;
            d.vmat[i] = d.vmat[ip] * v_factor;
            d.pmat[i] = d.pmat[ip];
            d.cmat[i] = d.cmat[ip];
            d.gmat[i] = d.gmat[ip];

            i  -= disp;
            ip -= disp;
        }
    });
}


void boundaryConditions(const Params& p, Data& d)
{
    constexpr std::array<Side, 2> X_pass = { Side::Left, Side::Right };
    constexpr std::array<Side, 2> Y_pass = { Side::Top, Side::Bottom };
    const std::array<Side, 2>& side_order = p.current_axis == Axis::X ? X_pass : Y_pass;
    for (Side side : side_order) {
        boundaryConditions(p, d, side);
    }
}


void euler_projection(const Params& p, Data& d, flt_t dt,
                      view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    const int s = p.s;
    Kokkos::parallel_for(iter(real_domain(p)),
    KOKKOS_LAMBDA(const int i) {
        flt_t dX = p.dx + dt * (d.ustar[i+s] - d.ustar[i]) * d.domain_mask[i];

        flt_t mask = d.domain_mask[i] * d.domain_mask[i+s];

        flt_t tmp_rho  = (dX * d.rho[i]             - mask * (advection_rho[i+s]  - advection_rho[i] )) / p.dx;
        flt_t tmp_urho = (dX * d.rho[i] * d.umat[i] - mask * (advection_urho[i+s] - advection_urho[i])) / p.dx;
        flt_t tmp_vrho = (dX * d.rho[i] * d.vmat[i] - mask * (advection_vrho[i+s] - advection_vrho[i])) / p.dx;
        flt_t tmp_Erho = (dX * d.rho[i] * d.Emat[i] - mask * (advection_Erho[i+s] - advection_Erho[i])) / p.dx;

        d.rho[i]  = tmp_rho;
        d.umat[i] = tmp_urho / tmp_rho;
        d.vmat[i] = tmp_vrho / tmp_rho;
        d.Emat[i] = tmp_Erho / tmp_rho;
    });
}


void advection_first_order(const Params& p, Data& d, flt_t dt,
                           view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    const int s = p.s;
    Kokkos::parallel_for(iter(real_domain_advection(p)),
    KOKKOS_LAMBDA(const int idx) {
        int i = idx;
        flt_t disp = dt * d.ustar[i];
        if (disp > 0) {
            i = idx - s;
        }

        disp *= d.domain_mask[i];  // Don't advect this cell if it isn't real

        advection_rho[idx]  = disp * (d.rho[i]            );
        advection_urho[idx] = disp * (d.rho[i] * d.umat[i]);
        advection_vrho[idx] = disp * (d.rho[i] * d.vmat[i]);
        advection_Erho[idx] = disp * (d.rho[i] * d.Emat[i]);
    });
}


KOKKOS_INLINE_FUNCTION flt_t slope_minmod(flt_t u_im, flt_t u_i, flt_t u_ip, flt_t r_m, flt_t r_p)
{
    flt_t D_u_p = r_p * (u_ip - u_i );
    flt_t D_u_m = r_m * (u_i  - u_im);
    flt_t s = std::copysign(flt_t(1), D_u_p);
    return s * std::max(flt_t(0), std::min(s * D_u_p, s * D_u_m));
}


void advection_second_order(const Params& p, Data& d, flt_t dt,
                           view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    const int s = p.s;
    Kokkos::parallel_for(iter(real_domain_advection(p)),
    KOKKOS_LAMBDA(const int idx) {
        int i = idx;
        flt_t disp = dt * d.ustar[i];
        flt_t Dx;
        if (disp > 0) {
            Dx = -(p.dx - dt * d.ustar[i-s]);
            i = idx - s;
        } else {
            Dx = p.dx + dt * d.ustar[i+s];
        }

        disp *= d.domain_mask[i];  // Don't advect this cell if it isn't real

        // Set the 2nd order contribution to 0 if any neighbouring cell isn't real
        flt_t mask = d.domain_mask[i-s] * d.domain_mask[i] * d.domain_mask[i+s];

        flt_t Dx_lm = p.dx + dt * (d.ustar[i]     - d.ustar[i-s]);
        flt_t Dx_l  = p.dx + dt * (d.ustar[i+s]   - d.ustar[i]  );
        flt_t Dx_lp = p.dx + dt * (d.ustar[i+2*s] - d.ustar[i+s]);

        flt_t r_m = (2 * Dx_l) / (Dx_l + Dx_lm);
        flt_t r_p = (2 * Dx_l) / (Dx_l + Dx_lp);

        flt_t slope_r  = slope_minmod(d.rho[i-s]              , d.rho[i]            , d.rho[i+s]              , r_m, r_p);
        flt_t slope_ur = slope_minmod(d.rho[i-s] * d.umat[i-s], d.rho[i] * d.umat[i], d.rho[i+s] * d.umat[i+s], r_m, r_p);
        flt_t slope_vr = slope_minmod(d.rho[i-s] * d.vmat[i-s], d.rho[i] * d.vmat[i], d.rho[i+s] * d.vmat[i+s], r_m, r_p);
        flt_t slope_Er = slope_minmod(d.rho[i-s] * d.Emat[i-s], d.rho[i] * d.Emat[i], d.rho[i+s] * d.Emat[i+s], r_m, r_p);

        flt_t length_factor = Dx / (2 * Dx_l) * mask;
        advection_rho[idx]  = disp * (d.rho[i]             - slope_r  * length_factor);
        advection_urho[idx] = disp * (d.rho[i] * d.umat[i] - slope_ur * length_factor);
        advection_vrho[idx] = disp * (d.rho[i] * d.vmat[i] - slope_vr * length_factor);
        advection_Erho[idx] = disp * (d.rho[i] * d.Emat[i] - slope_Er * length_factor);
    });
}


void projection_remap(const Params& p, Data& d, flt_t dt)
{
    if (p.projection == Projection::None) return;

    view& advection_rho  = d.work_array_1;
    view& advection_urho = d.work_array_2;
    view& advection_vrho = d.work_array_3;
    view& advection_Erho = d.work_array_4;

    if (p.projection == Projection::Euler) {
        advection_first_order(p, d, dt, advection_rho, advection_urho, advection_vrho, advection_Erho);
    }
    else if (p.projection == Projection::Euler_2nd) {
        advection_second_order(p, d, dt, advection_rho, advection_urho, advection_vrho, advection_Erho);
    }

    euler_projection(p, d, dt, advection_rho, advection_urho, advection_vrho, advection_Erho);
}


flt_t dtCFL(const Params& p, Data& d, flt_t dta)
{
    flt_t dt = INFINITY;
    flt_t dx = p.domain_size[0] / flt_t(p.nx);
    flt_t dy = p.domain_size[1] / flt_t(p.ny);

    if (p.cst_dt) {
        return p.Dt;
    }
    else if (p.projection != Projection::None) {
        Kokkos::parallel_reduce(iter(real_domain(p)),
        KOKKOS_LAMBDA(const int i, flt_t& dt_loop) {
            flt_t max_cx = std::max(std::abs(d.umat[i] + d.cmat[i]), std::abs(d.umat[i] - d.cmat[i])) * d.domain_mask[i];
            flt_t max_cy = std::max(std::abs(d.vmat[i] + d.cmat[i]), std::abs(d.vmat[i] - d.cmat[i])) * d.domain_mask[i];
            dt_loop = std::min(dt_loop, std::min(dx / max_cx, dy / max_cy));
        }, Kokkos::Min<flt_t>(dt));
    }
    else {
        Kokkos::parallel_reduce(iter(real_domain(p)),
        KOKKOS_LAMBDA(const int i, flt_t& dt_loop) {
            dt_loop = std::min(dt_loop, flt_t(1.) / (d.cmat[i] * d.domain_mask[i]));
        }, Kokkos::Min<flt_t>(dt));
        dt *= std::min(dx, dy);
    }

    if (!std::isfinite(dt) || dt <= 0)
        return dt;
    else if (dta == 0)
        return p.Dt != 0 ? p.Dt : p.cfl * dt;
    else
        return std::min(p.cfl * dt, flt_t(1.05) * dta);
}


bool step_checkpoint(const Params& p, const Data& d, HostData& hd, const char* step_label, int cycle, const char* axis)
{
    if (!p.compare) return false;

    d.deep_copy_to_mirror(hd);

    char buf[100];
    snprintf(buf, 100, "_%03d_%s", cycle, step_label);
    std::string step_file_name = std::string(p.output_file) + buf + (std::strlen(axis) > 0 ? "_" : "") + axis;

    bool is_different;
    try {
        is_different = compare_with_file(p, hd, step_file_name);
    } catch (std::exception& e) {
        std::cerr << "Error while comparing with file '" << step_file_name << "': " << e.what() << std::endl;
        is_different = true;
    }

    if (is_different) {
        std::string diff_file_name = step_file_name + "_diff";
        write_output(p, hd, diff_file_name.c_str());
        printf("Difference file written to %s\n", diff_file_name.c_str());
    }

    return is_different;
}


bool step_checkpoint(const Params& p, const Data& d, HostData& hd, const char* step_label, int cycle, Axis axis)
{
    switch (axis) {
    case Axis::X: return step_checkpoint(p, d, hd, step_label, cycle, "X");
    case Axis::Y: return step_checkpoint(p, d, hd, step_label, cycle, "Y");
    default:      return false;
    }
}


#define CHECK_STEP(label) if (step_checkpoint(p, d, hd, label, cycles, axis)) goto end_loop


std::tuple<flt_t, flt_t> conservation_vars(const Params& p, Data& d)
{
    flt_t total_mass = 0;
    flt_t total_energy = 0;

    if (p.projection == Projection::None) {
        int s = p.row_length;
        Kokkos::parallel_reduce(iter(real_domain(p)),
        KOKKOS_LAMBDA(const int i, flt_t& mass, flt_t& energy) {
            flt_t ds = (d.x[i+1] - d.x[i]) * (d.y[i+s] - d.y[i]);
            flt_t cell_mass = d.rho[i] * ds * d.domain_mask[i];
            flt_t cell_energy = cell_mass * d.Emat[i];
            mass += cell_mass;
            energy += cell_energy;
        }, Kokkos::Sum<flt_t>(total_mass), Kokkos::Sum<flt_t>(total_energy));
    }
    else {
        flt_t ds = p.dx * p.dx;
        Kokkos::parallel_reduce(iter(real_domain(p)),
        KOKKOS_LAMBDA(const int i, flt_t& mass, flt_t& energy) {
            flt_t cell_mass = d.rho[i] * d.domain_mask[i] * ds;
            flt_t cell_energy = cell_mass * d.Emat[i];
            mass += cell_mass;
            energy += cell_energy;
        }, Kokkos::Sum<flt_t>(total_mass), Kokkos::Sum<flt_t>(total_energy));
    }

    return std::make_tuple(total_mass, total_energy);
}


std::tuple<double, flt_t, int> time_loop(Params& p, Data& d, HostData& hd)
{
    int cycles = 0;
    flt_t t = 0., prev_dt = 0., next_dt = 0.;

    auto time_loop_start = std::chrono::steady_clock::now();

    p.update_axis(Axis::X);

    update_EOS(p, d);  // Finalize the initialisation by calling the EOS

    flt_t initial_mass, initial_energy;
    if (p.verbose <= 1) {
        std::tie(initial_mass, initial_energy) = conservation_vars(p, d);
    }

    while (t < p.max_time && cycles < p.max_cycles) {
        TIC(); next_dt = dtCFL(p, d, prev_dt);  TAC("dtCFL");

        if (!std::isfinite(next_dt) || next_dt <= 0.) {
            printf("Invalid dt at cycle %d: %f\n", cycles, next_dt);
            Kokkos::finalize();
            exit(1);
        }

        if (cycles == 0) {
            prev_dt = next_dt;
        }

        for (auto [axis, dt_factor] : p.split_axes(cycles)) {
            p.update_axis(axis);

            TIC(); update_EOS(p, d);                            TAC("update_EOS");         CHECK_STEP("update_EOS");
            TIC(); boundaryConditions(p, d);                    TAC("boundaryConditions"); CHECK_STEP("boundaryConditions");
            TIC(); numericalFluxes(p, d, prev_dt * dt_factor);  TAC("numericalFluxes");    CHECK_STEP("numericalFluxes");
            TIC(); cellUpdate(p, d, prev_dt * dt_factor);       TAC("cellUpdate");         CHECK_STEP("cellUpdate");
            TIC(); projection_remap(p, d, prev_dt * dt_factor); TAC("euler_proj");         CHECK_STEP("projection_remap");
        }

        if (p.verbose <= 1) {
            auto [current_mass, current_energy] = conservation_vars(p, d);
            flt_t delta_mass   = std::abs(initial_mass   - current_mass)   / initial_mass   * 100;
            flt_t delta_energy = std::abs(initial_energy - current_energy) / initial_energy * 100;
            printf("Cycle = %4d, dt = %.18f, t = %.18f, |ΔM| = %8.6f%%, |ΔE| = %8.6f%%\n",
                   cycles, prev_dt, t, delta_mass, delta_energy);
        }

        t += prev_dt;
        prev_dt = next_dt;
        cycles++;
    }

end_loop:

    auto time_loop_end = std::chrono::steady_clock::now();

    double loop_time = std::chrono::duration<double>(time_loop_end - time_loop_start).count();
    double grind_time = loop_time / (static_cast<double>(cycles) * p.nx * p.ny) * 1e6;

    if (p.verbose < 4) {
        printf("\n");
        printf("Time:       %.4g seconds\n", loop_time);
        printf("Grind time: %.4g µs/cell/cycle\n", grind_time);
        printf("Cells/sec:  %.4g Mega cells/sec\n", 1. / grind_time);
        printf("Cycles:     %d\n", cycles);
        printf("Final dt:   %.18f\n\n", next_dt);
    }

    return std::make_tuple(grind_time, next_dt, cycles);
}


bool armon(Params& params)
{
    Data data(params.nb_cells, "Armon");
    HostData host_data = data.as_mirror();

    TIC(); init_test(params, data); TAC("init_test");
    double grind_time;
    std::tie(grind_time, std::ignore, std::ignore) = time_loop(params, data, host_data);

    data.deep_copy_to_mirror(host_data);

    if (params.write_output) {
        write_output(params, host_data);
    }

    if (params.write_throughput) {
        FILE* grind_time_file = fopen("cell_throughput", "w");
        fprintf(grind_time_file, "%f", 1. / grind_time);
        fclose(grind_time_file);
    }

    if (params.verbose < 3) {
        double total_time = time_contribution["TOTAL"];
        time_contribution.erase(time_contribution.find("TOTAL"));

        printf("Total time for each step:\n");
        for (const auto& [label, time] : time_contribution) {
            printf(" - %-20s %10.5f ms (%5.2f%%)\n", label.c_str(), time * 1e3, time / total_time * 100);
        }
    }

    return true;
}
