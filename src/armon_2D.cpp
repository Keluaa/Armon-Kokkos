
#include "armon_2D.h"

#include "io.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <string>
#include <array>

#if USE_NVTX == 1
#include <nvtx3/nvToolsExt.h>
#include <string_view>

auto nvtxAttribs(const char* message)
{
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = std::hash<std::string_view>{}(std::string_view(message));
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = message;
    return attr;
}

static nvtxDomainHandle_t ARMON_DOMAIN = nullptr;

#define BEGIN_RANGE(name)                      \
    auto attr_r_ ## name = nvtxAttribs(#name); \
    auto range_hdl_ ## name = nvtxDomainRangeStartEx(ARMON_DOMAIN, &(attr_r_ ## name))

#define END_RANGE(name) \
    nvtxRangeEnd(range_hdl_ ## name)

void init_nvtx()
{
    if (ARMON_DOMAIN == nullptr) {
        ARMON_DOMAIN = nvtxDomainCreateA("Armon");
    }
}

#else
void init_nvtx() {}
#define BEGIN_RANGE(name) do {} while (false)
#define END_RANGE(name) do {} while (false)
#endif


template<typename Functor>
void parallel_kernel(const Range& range, const Functor& functor)
{
#if USE_SIMD_KERNELS
    // Kokkos is unable to correctly detect SIMD register sizes for any CPU: we must do it ourselves.
    constexpr unsigned int V = PREFERRED_SIMD_SIZE / sizeof(flt_t);
    Kokkos::parallel_for(iter_simd(range, V),
    KOKKOS_LAMBDA(const Team_t& team) {
        const Idx team_idx_size = team.team_size() * V;
        const Idx team_i = range.start + team.league_rank() * team_idx_size;
        const auto team_threads = Kokkos::TeamThreadRange(team, team.team_size());
        Kokkos::parallel_for(team_threads, [&](Idx thread_idx) {
            const Idx thread_i = team_i + thread_idx * V;
            const Idx thread_end = Kokkos::min(thread_i + V, range.end);
            const auto thread_vectors = Kokkos::ThreadVectorRange(team, thread_i, thread_end);
            Kokkos::parallel_for(thread_vectors, functor);
        });
    });
#else
    Kokkos::parallel_for(iter(range), functor);
#endif  // USE_SIMD_KERNELS
}


template<typename Functor, typename Reducer>
void parallel_reduce_kernel(const Range& range, const Functor& functor, const Reducer& global_reducer)
{
#if USE_SIMD_KERNELS
    constexpr unsigned int V = PREFERRED_SIMD_SIZE / sizeof(flt_t);

    using R_ref = decltype(global_reducer.reference());
    using R_val = typename Reducer::value_type;

    // Hierarchical parallelism => Hierarchical reduction: one reducer per loop, each accumulating into the upper one

    Kokkos::parallel_reduce(iter_simd(range, V),
    KOKKOS_LAMBDA(const Team_t& team, R_ref result) {
        const Idx team_idx_size = team.team_size() * V;
        const Idx team_i = range.start + team.league_rank() * team_idx_size;
        const auto team_threads = Kokkos::TeamThreadRange(team, team.team_size());

        R_val team_result;
        const auto team_reducer = Reducer(team_result);
        team_reducer.init(team_result);

        Kokkos::parallel_reduce(team_threads, [&](Idx thread_idx, R_ref threads_result) {
            const Idx thread_i = team_i + thread_idx * V;
            const Idx thread_end = Kokkos::min(thread_i + V, range.end);
            const auto thread_vectors = Kokkos::ThreadVectorRange(team, thread_i, thread_end);

            R_val thread_result;
            const auto thread_reducer = Reducer(thread_result);
            thread_reducer.init(thread_result);

            Kokkos::parallel_reduce(thread_vectors, functor, thread_reducer);

            thread_reducer.join(threads_result, thread_result);
        }, team_reducer);

        if (team.team_rank() == 0) {
            team_reducer.join(result, team_result);  // Accumulate once per team
        }
    }, global_reducer);
#else
    Kokkos::parallel_reduce(iter(range), functor, global_reducer);
#endif  // USE_SIMD_KERNELS
}


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
    UNPACK_FIELDS(d, rho, cmat, pmat, ustar, pstar);
    UNPACK_FIELDS(p, s);

    parallel_kernel(real_domain_fluxes(p),
    KOKKOS_LAMBDA(const Idx i) {
        auto [ustar_i, pstar_i] = acoustic_Godunov(
            rho[i], rho[i-s], cmat[i], cmat[i-s],
              u[i],   u[i-s], pmat[i], pmat[i-s]);
        ustar[i] = ustar_i;
        pstar[i] = pstar_i;
    });
}


template<Limiter L>
void acoustic_GAD(const Params& p, Data& d, flt_t dt, const view& u)
{
    UNPACK_FIELDS(d, rho, cmat, pmat, ustar, pstar);
    UNPACK_FIELDS(p, s, dx);

    parallel_kernel(real_domain_fluxes(p),
    KOKKOS_LAMBDA(const Idx i) {
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
    UNPACK_FIELDS(d, rho, Emat, umat, vmat, pmat, cmat, gmat);

    parallel_kernel(real_domain(p),
    KOKKOS_LAMBDA(const Idx i) {
        flt_t e = Emat[i] - flt_t(0.5) * (Kokkos::pow(umat[i], flt_t(2)) + Kokkos::pow(vmat[i], flt_t(2)));
        pmat[i] = (gamma - 1) * rho[i] * e;
        cmat[i] = Kokkos::sqrt(gamma * pmat[i] / rho[i]);
        gmat[i] = (1 + gamma) / 2;
    });
}


void bizarriumEOS(const Params& p, Data& d)
{
    const flt_t rho0 = 1e4, K0 = 1e11, Cv0 = 1e3, T0 = 300, eps0 = 0;
    const flt_t G0 = 1.5, s = 1.5, q = -42080895./14941154., r = 727668333./149411540.;
    UNPACK_FIELDS(d, rho, Emat, umat, vmat, pmat, cmat, gmat);

    parallel_kernel(real_domain(p),
    KOKKOS_LAMBDA(const Idx i) {
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
    view& u = p.current_axis == Axis::X ? d.umat : d.vmat;
    view& x = p.current_axis == Axis::X ? d.x : d.y;

    UNPACK_FIELDS(d, rho, Emat, ustar, pstar, domain_mask);
    UNPACK_FIELDS(p, s, dx);

    parallel_kernel(real_domain(p),
    KOKKOS_LAMBDA(const Idx i) {
        flt_t mask = domain_mask[i];
        flt_t dm = rho[i] * dx;
        rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]) * mask);
        u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             ) * mask;
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]) * mask;
    });

    if (p.projection == Projection::None) {
        parallel_kernel(real_domain(p),
        KOKKOS_LAMBDA(const Idx i) {
            x[i] += dt * ustar[i];
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

    UNPACK_FIELDS(d, x, y, rho, Emat, umat, vmat, domain_mask, pmat, cmat, ustar, pstar);
    UNPACK_FIELDS(p, row_length, nb_ghosts, nx, ny, test);

    parallel_kernel(all_cells(p),
    KOKKOS_LAMBDA(const Idx i) {
        int ix = (static_cast<int>(i) % row_length) - nb_ghosts;
        int iy = (static_cast<int>(i) / row_length) - nb_ghosts;

        x[i] = flt_t(ix) / flt_t(nx) * sx + ox;
        y[i] = flt_t(iy) / flt_t(ny) * sy + oy;

        flt_t x_mid = x[i] + sx / (2 * nx);
        flt_t y_mid = y[i] + sy / (2 * ny);

        if (test_region_high(test, x_mid, y_mid)) {
            rho[i] = tp.high_rho;
            Emat[i] = tp.high_p / ((tp.gamma - flt_t(1)) * rho[i]);
            umat[i] = tp.high_u;
            vmat[i] = tp.high_v;
        } else {
            rho[i] = tp.low_rho;
            Emat[i] = tp.low_p / ((tp.gamma - flt_t(1)) * rho[i]);
            umat[i] = tp.low_u;
            vmat[i] = tp.low_v;
        }

        if (one_more_ring) {
            domain_mask[i] = (
                   ((-r <= ix) && (ix < nx+r) && (-r <= iy) && (iy < ny+r))  // Include as well a ring of ghost cells...
                && (( 0 <= ix) && (ix < nx)   || (0  <= iy) && (iy < ny)  )  // ...while excluding the corners of the subdomain
            );
        } else {
            domain_mask[i] = (0 <= ix && ix < nx && 0 <= iy && iy < ny);
        }

        // Set to zero to make sure no non-initialized values changes the result
        pmat[i] = 0;
        cmat[i] = 1;  // Set to 1 as a max speed of 0 will create NaNs
        ustar[i] = 0;
        pstar[i] = 0;
    });
}


void boundaryConditions(const Params& p, Data& d, Side side)
{
    int stride = 1, disp = 1, i_start, loop_range;

    auto factors = boundaryCondition(p.test, side);
    flt_t u_factor = factors[0];
    flt_t v_factor = factors[1];

    switch (side) {
    case Side::Left:
        stride = p.row_length;
        i_start = index_1D(p, -1, 0);
        loop_range = p.ny;
        disp = 1;
        break;

    case Side::Right:
        stride = p.row_length;
        i_start = index_1D(p, p.nx, 0);
        loop_range = p.ny;
        disp = -1;
        break;

    case Side::Top:
        stride = 1;
        i_start = index_1D(p, 0, p.ny);
        loop_range = p.nx;
        disp = -p.row_length;
        break;

    case Side::Bottom:
        stride = 1;
        i_start = index_1D(p, 0, -1);
        loop_range = p.nx;
        disp = p.row_length;
        break;
    }

    UNPACK_FIELDS(d, rho, umat, vmat, pmat, cmat, gmat);
    UNPACK_FIELDS(p, stencil_width);

    parallel_kernel(zero_to(loop_range - 1),
    KOKKOS_LAMBDA(const Idx idx) {
        Idx i = idx * stride + i_start;
        Idx ip = i + disp;

        for (Idx w = 0; w < stencil_width; w++) {
            rho[i]  = rho[ip];
            umat[i] = umat[ip] * u_factor;
            vmat[i] = vmat[ip] * v_factor;
            pmat[i] = pmat[ip];
            cmat[i] = cmat[ip];
            gmat[i] = gmat[ip];

            i  -= disp;
            ip += disp;
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
                      const view& advection_rho, const view& advection_urho,
                      const view& advection_vrho, const view& advection_Erho)
{
    UNPACK_FIELDS(d, rho, umat, vmat, Emat, ustar, domain_mask);
    UNPACK_FIELDS(p, dx, s);

    parallel_kernel(real_domain(p),
    KOKKOS_LAMBDA(const Idx i) {
        flt_t mask = domain_mask[i];
        flt_t dX = dx + dt * (ustar[i+s] - ustar[i]) * mask;

        flt_t tmp_rho  = (dX * rho[i]           - mask * (advection_rho[i+s]  - advection_rho[i] )) / dx;
        flt_t tmp_urho = (dX * rho[i] * umat[i] - mask * (advection_urho[i+s] - advection_urho[i])) / dx;
        flt_t tmp_vrho = (dX * rho[i] * vmat[i] - mask * (advection_vrho[i+s] - advection_vrho[i])) / dx;
        flt_t tmp_Erho = (dX * rho[i] * Emat[i] - mask * (advection_Erho[i+s] - advection_Erho[i])) / dx;

        rho[i]  = tmp_rho;
        umat[i] = tmp_urho / tmp_rho;
        vmat[i] = tmp_vrho / tmp_rho;
        Emat[i] = tmp_Erho / tmp_rho;
    });
}


void advection_first_order(const Params& p, Data& d, flt_t dt,
                           view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    UNPACK_FIELDS(d, rho, umat, vmat, Emat, ustar);
    UNPACK_FIELDS(p, s);

    parallel_kernel(real_domain_advection(p),
    KOKKOS_LAMBDA(const Idx idx) {
        Idx i = idx;
        flt_t disp = dt * ustar[i];
        if (disp > 0) {
            i = idx - s;
        }

        advection_rho[idx]  = disp * (rho[i]          );
        advection_urho[idx] = disp * (rho[i] * umat[i]);
        advection_vrho[idx] = disp * (rho[i] * vmat[i]);
        advection_Erho[idx] = disp * (rho[i] * Emat[i]);
    });
}


KOKKOS_INLINE_FUNCTION flt_t slope_minmod(flt_t u_im, flt_t u_i, flt_t u_ip, flt_t r_m, flt_t r_p)
{
    flt_t D_u_p = r_p * (u_ip - u_i );
    flt_t D_u_m = r_m * (u_i  - u_im);
    flt_t s = Kokkos::copysign(flt_t(1), D_u_p);
    return s * Kokkos::max(flt_t(0), Kokkos::min(s * D_u_p, s * D_u_m));
}


void advection_second_order(const Params& p, Data& d, flt_t dt,
                            view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    UNPACK_FIELDS(d, rho, umat, vmat, Emat, ustar);
    UNPACK_FIELDS(p, s, dx);

    parallel_kernel(real_domain_advection(p),
    KOKKOS_LAMBDA(const Idx idx) {
        int i = static_cast<int>(idx);
        flt_t disp = dt * ustar[i];
        flt_t Dx;
        if (disp > 0) {
            Dx = -(dx - dt * ustar[i-s]);
            i = static_cast<int>(idx) - s;
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
        advection_rho[idx]  = disp * (rho[i]           - slope_r  * length_factor);
        advection_urho[idx] = disp * (rho[i] * umat[i] - slope_ur * length_factor);
        advection_vrho[idx] = disp * (rho[i] * vmat[i] - slope_vr * length_factor);
        advection_Erho[idx] = disp * (rho[i] * Emat[i] - slope_Er * length_factor);
    });
}


void projection_remap(const Params& p, Data& d, flt_t dt)
{
    if (p.projection == Projection::None) return;

    view& advection_rho  = d.work_array_1;
    view& advection_urho = d.work_array_2;
    view& advection_vrho = d.work_array_3;
    view& advection_Erho = d.work_array_4;

    BEGIN_RANGE(advection);
    if (p.projection == Projection::Euler) {
        advection_first_order(p, d, dt, advection_rho, advection_urho, advection_vrho, advection_Erho);
    }
    else if (p.projection == Projection::Euler_2nd) {
        advection_second_order(p, d, dt, advection_rho, advection_urho, advection_vrho, advection_Erho);
    }
    END_RANGE(advection);

    BEGIN_RANGE(projection);
    euler_projection(p, d, dt, advection_rho, advection_urho, advection_vrho, advection_Erho);
    END_RANGE(projection);
}


flt_t dtCFL(const Params& p, Data& d, flt_t dta)
{
    flt_t dt = INFINITY;
    flt_t dx = p.domain_size[0] / flt_t(p.nx);
    flt_t dy = p.domain_size[1] / flt_t(p.ny);

    if (p.cst_dt) {
        return p.Dt;
    } else if (p.projection != Projection::None) {
        UNPACK_FIELDS(d, umat, vmat, cmat, domain_mask);
        parallel_reduce_kernel(real_domain(p),
        KOKKOS_LAMBDA(const Idx i, flt_t &dt_loop) {
            flt_t max_cx = Kokkos::max(Kokkos::abs(umat[i] + cmat[i]), Kokkos::abs(umat[i] - cmat[i])) * domain_mask[i];
            flt_t max_cy = Kokkos::max(Kokkos::abs(vmat[i] + cmat[i]), Kokkos::abs(vmat[i] - cmat[i])) * domain_mask[i];
            dt_loop = Kokkos::min(dt_loop, Kokkos::min(dx / max_cx, dy / max_cy));
        }, Kokkos::Min<flt_t>(dt));
    } else {
        UNPACK_FIELDS(d, cmat, domain_mask);
        parallel_reduce_kernel(real_domain(p),
        KOKKOS_LAMBDA(const Idx i, flt_t &dt_loop) {
            dt_loop = Kokkos::min(dt_loop, flt_t(1.) / (cmat[i] * domain_mask[i]));
        }, Kokkos::Min<flt_t>(dt));
        dt *= Kokkos::min(dx, dy);
    }

    if (!is_ieee754_finite(dt) || dt <= 0)
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


#define CHECK_STEP(label) if (step_checkpoint(p, d, hd, label, cycles, axis)) return true


std::tuple<flt_t, flt_t> conservation_vars(const Params& p, Data& d)
{
    flt_t total_mass = 0;
    flt_t total_energy = 0;

    if (p.projection == Projection::None) {
        int s = p.row_length;
        UNPACK_FIELDS(d, x, y, rho, Emat, domain_mask);
        Kokkos::parallel_reduce(iter(real_domain(p)),
        KOKKOS_LAMBDA(const Idx i, flt_t& mass, flt_t& energy) {
            flt_t ds = (x[i+1] - x[i]) * (y[i+s] - y[i]);
            flt_t cell_mass = rho[i] * ds * domain_mask[i];
            flt_t cell_energy = cell_mass * Emat[i];
            mass += cell_mass;
            energy += cell_energy;
        }, Kokkos::Sum<flt_t>(total_mass), Kokkos::Sum<flt_t>(total_energy));
    } else {
        flt_t ds = p.dx * p.dx;
        UNPACK_FIELDS(d, x, y, rho, Emat, domain_mask);
        Kokkos::parallel_reduce(iter(real_domain(p)),
        KOKKOS_LAMBDA(const Idx i, flt_t& mass, flt_t& energy) {
            flt_t cell_mass = rho[i] * domain_mask[i] * ds;
            flt_t cell_energy = cell_mass * Emat[i];
            mass += cell_mass;
            energy += cell_energy;
        }, Kokkos::Sum<flt_t>(total_mass), Kokkos::Sum<flt_t>(total_energy));
    }

    return std::make_tuple(total_mass, total_energy);
}


bool solver_cycle(Params& p, Data& d, HostData& hd, int cycles, flt_t prev_dt)
{
    for (auto [axis, dt_factor] : p.split_axes(cycles)) {
        p.update_axis(axis);

        BEGIN_RANGE(axis);
        BEGIN_RANGE(EOS);    TIC(); update_EOS(p, d);                            TAC("update_EOS");         END_RANGE(EOS);    CHECK_STEP("update_EOS");
        BEGIN_RANGE(BC);     TIC(); boundaryConditions(p, d);                    TAC("boundaryConditions"); END_RANGE(BC);     CHECK_STEP("boundaryConditions");
        BEGIN_RANGE(fluxes); TIC(); numericalFluxes(p, d, prev_dt * dt_factor);  TAC("numericalFluxes");    END_RANGE(fluxes); CHECK_STEP("numericalFluxes");
        BEGIN_RANGE(update); TIC(); cellUpdate(p, d, prev_dt * dt_factor);       TAC("cellUpdate");         END_RANGE(update); CHECK_STEP("cellUpdate");
        BEGIN_RANGE(remap);  TIC(); projection_remap(p, d, prev_dt * dt_factor); TAC("euler_proj");         END_RANGE(remap);  CHECK_STEP("projection_remap");
        END_RANGE(axis);
    }

    return false;
}


std::tuple<double, flt_t, int> time_loop(Params& p, Data& d, HostData& hd)
{
    int cycles = 0;
    flt_t t = 0., prev_dt = 0., next_dt = 0.;

    auto time_loop_start = std::chrono::steady_clock::now();

    p.update_axis(Axis::X);

    BEGIN_RANGE(EOS_init);
    update_EOS(p, d);  // Finalize the initialisation by calling the EOS
    END_RANGE(EOS_init);
    if (step_checkpoint(p, d, hd, "update_EOS_init", 0, p.current_axis)) goto end_loop;

    flt_t initial_mass, initial_energy;
    if (p.verbose <= 1) {
        std::tie(initial_mass, initial_energy) = conservation_vars(p, d);
    }

    while (t < p.max_time && cycles < p.max_cycles) {
        BEGIN_RANGE(cycle);
        BEGIN_RANGE(time_step);
        TIC(); next_dt = dtCFL(p, d, prev_dt);  TAC("dtCFL");
        END_RANGE(time_step);

        if (!is_ieee754_finite(next_dt) || next_dt <= 0.) {
            printf("Invalid dt at cycle %d: %f\n", cycles, next_dt);
            Kokkos::finalize();
            exit(1);
        }

        if (cycles == 0) {
            prev_dt = next_dt;
        }

        if (solver_cycle(p, d, hd, cycles, prev_dt)) goto end_loop;

        if (p.verbose <= 1) {
            auto [current_mass, current_energy] = conservation_vars(p, d);
            flt_t delta_mass   = std::abs(initial_mass   - current_mass)   / initial_mass   * 100;
            flt_t delta_energy = std::abs(initial_energy - current_energy) / initial_energy * 100;
            printf("Cycle = %4d, dt = %.18f, t = %.18f, |ΔM| = %#8.6g%%, |ΔE| = %#8.6g%%\n",
                   cycles, prev_dt, t, delta_mass, delta_energy);
        }

        t += prev_dt;
        prev_dt = next_dt;
        cycles++;

        END_RANGE(cycle);
    }

    {
        BEGIN_RANGE(last_fence);
        Kokkos::fence("last_fence");
        END_RANGE(last_fence);
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
    time_contribution.clear();
    init_nvtx();

    BEGIN_RANGE(init);
    BEGIN_RANGE(alloc);
    Data data(params.nb_cells, "Armon");
    HostData host_data = (params.compare || params.write_output) ? data.as_mirror() : HostData{0};
    END_RANGE(alloc);

    BEGIN_RANGE(init_test);
    TIC(); init_test(params, data); TAC("init_test");
    END_RANGE(init_test);
    END_RANGE(init);

    double grind_time;
    std::tie(grind_time, std::ignore, std::ignore) = time_loop(params, data, host_data);

    if (params.write_output) {
        data.deep_copy_to_mirror(host_data);
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
