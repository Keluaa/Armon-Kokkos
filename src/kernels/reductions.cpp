
#include "kernels.h"
#include "parallel_kernels.h"
#include "double_reducer.h"
#include "utils.h"


extern "C"
flt_t dt_CFL(const Range& range, const InnerRange1D& inner_range, flt_t dx, flt_t dy,
             const view& umat, const view& vmat, const view& cmat, const mask_view& domain_mask)
KERNEL_TRY {
    flt_t dt = INFINITY;

    CHECK_VIEW_LABELS(umat, vmat, cmat, domain_mask);

    parallel_reduce_kernel(range, KOKKOS_LAMBDA(const UIdx lin_i, flt_t& dt_loop) {
        Idx i = inner_range.scale_index(lin_i);
        flt_t max_cx = Kokkos::max(Kokkos::abs(umat[i] + cmat[i]), Kokkos::abs(umat[i] - cmat[i])) * domain_mask[i];
        flt_t max_cy = Kokkos::max(Kokkos::abs(vmat[i] + cmat[i]), Kokkos::abs(vmat[i] - cmat[i])) * domain_mask[i];
        dt_loop = Kokkos::min(dt_loop, Kokkos::min(dx / max_cx, dy / max_cy));
    }, Kokkos::Min<flt_t>(dt));

    return dt;
} KERNEL_CATCH


extern "C"
void conservation_vars(const Range& range, const InnerRange1D& inner_range, flt_t dx,
                       const view& rho, const view& Emat, const mask_view& domain_mask,
                       flt_t& total_mass, flt_t& total_energy)
KERNEL_TRY {
    auto reducer_tuple = std::make_tuple(total_mass, total_energy);
    auto reducer = DoubleReducer<Kokkos::Sum<flt_t>>(reducer_tuple);
    using Reducer_val = decltype(reducer)::value_type;

    CHECK_VIEW_LABELS(rho, Emat, domain_mask);

    flt_t ds = dx * dx;

    parallel_reduce_kernel(range, KOKKOS_LAMBDA(const UIdx lin_i, Reducer_val& mass_and_energy) {
        Idx i = inner_range.scale_index(lin_i);
        flt_t cell_mass = rho[i] * domain_mask[i] * ds;
        flt_t cell_energy = cell_mass * Emat[i];
        std::get<0>(mass_and_energy) += cell_mass;
        std::get<1>(mass_and_energy) += cell_energy;
    }, reducer);

    std::tie(total_mass, total_energy) = reducer_tuple;
} KERNEL_CATCH
