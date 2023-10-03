
#include "armon_2D.h"

#include "io.h"
#include "utils.h"
#include "kernels.h"
#include "test_cases.h"
#include "ranges.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <string>
#include <array>


// Solver profiling
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


// Solver step time tracking
std::map<std::string, double> time_contribution;
#define CAT(a, b) a##b
#define TIC_IMPL(line_nb) auto CAT(tic_, line_nb) = std::chrono::steady_clock::now()
#define TAC_IMPL(label, line_nb)                                \
    auto CAT(tac_, line_nb) = std::chrono::steady_clock::now(); \
    double CAT(expr_time_, line_nb) = std::chrono::duration<double>(CAT(tac_, line_nb) - CAT(tic_, line_nb)).count(); \
    time_contribution[label]   += CAT(expr_time_, line_nb);     \
    time_contribution["TOTAL"] += CAT(expr_time_, line_nb)
#define TIC() TIC_IMPL(__LINE__)
#define TAC(label) TAC_IMPL(label, __LINE__)


void numerical_fluxes(const Params& p, Data& d, flt_t dt)
{
    const view& u = p.current_axis == Axis::X ? d.umat : d.vmat;
    auto [range, inner_range] = domain_fluxes(p).iter2D();

    switch (p.riemann) {
    case Riemann::Acoustic:
        switch (p.scheme) {
        case Scheme::Godunov: return acoustic(range, inner_range, p.s, d.rho, u, d.cmat, d.pmat, d.ustar, d.pstar);
        case Scheme::GAD:     return acoustic_GAD(range, inner_range, p.s, p.dx, dt, d.rho, u, d.cmat, d.pmat, d.ustar, d.pstar, p.limiter);
        }
    }
}


void update_EOS(const Params& p, Data& d)
{
    auto [range, inner_range] = real_domain(p).iter2D();
    switch (p.test) {
    case Test::Sod:
    case Test::Sod_y:
    case Test::Sod_circ:
    case Test::Sedov:
    {
        const flt_t gamma = 1.4;
        perfect_gas_EOS(range, inner_range, gamma, d.rho, d.umat, d.vmat, d.Emat, d.cmat, d.pmat, d.gmat);
        break;
    }
    case Test::Bizarrium:
        bizarrium_EOS(range, inner_range, d.rho, d.umat, d.vmat, d.Emat, d.cmat, d.pmat, d.gmat);
        break;
    }
}


void cell_update(const Params& p, Data& d, flt_t dt)
{
    auto [range, inner_range] = domain_cell_update(p).iter2D();
    view& u = p.current_axis == Axis::X ? d.umat : d.vmat;
    cell_update(range, inner_range, p.s, p.dx, dt, d.ustar, d.pstar, d.Emat, d.rho, u);
}


void init_test(const Params& p, Data& d)
{
    auto [range, inner_range] = complete_domain(p).iter1D();
    init_test(range, inner_range,
              p.row_length, p.nb_ghosts,
              p.nx, p.ny, p.nx, p.ny, 0, 0,
              p.domain_size[0], p.domain_size[1], p.domain_origin[0], p.domain_origin[1],
              d.x, d.y, d.rho, d.Emat, d.umat, d.vmat,
              d.domain_mask, d.pmat, d.cmat, d.ustar, d.pstar,
              p.test, false);
}


void boundary_conditions(const Params& p, Data& d, Side side)
{
    int disp;
    auto domain_range = boundary_conditions_domain(p, side, disp);
    auto [u_factor, v_factor] = p.test_case->boundaryCondition(side);
    auto [range, inner_range] = domain_range.iter1D();
    boundary_conditions(range, inner_range,
                        disp, p.stencil_width,
                        u_factor, v_factor,
                        d.rho, d.umat, d.vmat, d.pmat, d.cmat, d.gmat, d.Emat);
}


void boundary_conditions(const Params& p, Data& d)
{
    constexpr std::array<Side, 2> X_pass = { Side::Left, Side::Right };
    constexpr std::array<Side, 2> Y_pass = { Side::Top, Side::Bottom };
    const std::array<Side, 2>& side_order = p.current_axis == Axis::X ? X_pass : Y_pass;
    for (Side side : side_order) {
        boundary_conditions(p, d, side);
    }
}


void euler_projection(const Params& p, Data& d, flt_t dt,
                      const view& advection_rho, const view& advection_urho,
                      const view& advection_vrho, const view& advection_Erho)
{
    auto [range, inner_range] = real_domain(p).iter2D();
    euler_projection(range, inner_range,
                     p.s, p.dx, dt,
                     d.ustar, d.rho, d.umat, d.vmat, d.Emat,
                     advection_rho, advection_urho, advection_vrho, advection_Erho);
}


void advection_first_order(const Params& p, Data& d, flt_t dt,
                           view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    auto [range, inner_range] = domain_advection(p).iter2D();
    advection_first_order(range, inner_range,
                          p.s, dt,
                          d.ustar, d.rho, d.umat, d.vmat, d.Emat,
                          advection_rho, advection_urho, advection_vrho, advection_Erho);
}


void advection_second_order(const Params& p, Data& d, flt_t dt,
                            view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
{
    auto [range, inner_range] = domain_advection(p).iter2D();
    advection_second_order(range, inner_range,
                           p.s, p.dx, dt,
                           d.ustar, d.rho, d.umat, d.vmat, d.Emat,
                           advection_rho, advection_urho, advection_vrho, advection_Erho);
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


flt_t dt_CFL(const Params& p, Data& d, flt_t dta)
{
    if (p.cst_dt) {
        return p.Dt;
    }

    flt_t dx = p.domain_size[0] / flt_t(p.nx);
    flt_t dy = p.domain_size[1] / flt_t(p.ny);
    auto [range, inner_range] = real_domain(p).iter1D();
    flt_t dt = dt_CFL(range, inner_range, dx, dy,
                      d.umat, d.vmat, d.cmat, d.domain_mask);

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


std::tuple<flt_t, flt_t> conservation_vars(const Params& p, Data& d)
{
    flt_t total_mass = 0;
    flt_t total_energy = 0;

    auto [range, inner_range] = real_domain(p).iter1D();
    conservation_vars(range, inner_range, p.dx,
                      d.rho, d.Emat, d.domain_mask,
                      total_mass, total_energy);

    return std::make_tuple(total_mass, total_energy);
}


#define CHECK_STEP(label) if (step_checkpoint(p, d, hd, label, cycles, axis)) return true


bool solver_cycle(Params& p, Data& d, HostData& hd, int cycles, flt_t prev_dt)
{
    for (auto [axis, dt_factor] : p.split_axes(cycles)) {
        p.update_axis(axis);

        BEGIN_RANGE(axis);
        BEGIN_RANGE(EOS);    TIC(); update_EOS(p, d);                            TAC("update_EOS");         END_RANGE(EOS);    CHECK_STEP("update_EOS");
        BEGIN_RANGE(BC);     TIC(); boundary_conditions(p, d);                    TAC("boundary_conditions"); END_RANGE(BC);     CHECK_STEP("boundary_conditions");
        BEGIN_RANGE(fluxes); TIC(); numerical_fluxes(p, d, prev_dt * dt_factor); TAC("numerical_fluxes");    END_RANGE(fluxes); CHECK_STEP("numerical_fluxes");
        BEGIN_RANGE(update); TIC(); cell_update(p, d, prev_dt * dt_factor);       TAC("cell_update");         END_RANGE(update); CHECK_STEP("cell_update");
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
        TIC(); next_dt = dt_CFL(p, d, prev_dt);  TAC("dt_CFL");
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
    Data data(params.nb_cells, "Armon_");
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
