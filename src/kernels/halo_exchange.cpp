
#include <Kokkos_Core.hpp>

#include "kernels.h"
#include "parallel_kernels.h"


extern "C"
void boundary_conditions(const Range& range, const InnerRange1D& inner_range,
                         Idx disp, Idx stencil_width,
                         flt_t u_factor, flt_t v_factor,
                         view& rho, view& umat, view& vmat, view& pmat, view& cmat, view& gmat, view& Emat)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = inner_range.scale_index(lin_i);
        Idx ip = i + disp;

        //   ghosts | real
        //      i   |  ip
        // ...3 2 1 | 1 2 3...  =>  iterate the cells from the border and outwards
        for (Idx w = 0; w < stencil_width; w++) {
            rho[i]  = rho[ip];
            umat[i] = umat[ip] * u_factor;
            vmat[i] = vmat[ip] * v_factor;
            pmat[i] = pmat[ip];
            cmat[i] = cmat[ip];
            gmat[i] = gmat[ip];
            Emat[i] = Emat[ip];

            i  -= disp;
            ip += disp;
        }
    });
}


extern "C"
void read_border_array(const Range& range, const InnerRange1D& inner_range,
                       Idx nghost, Idx side_length,
                       const view& rho, const view& umat, const view& vmat, const view& pmat,
                       const view& cmat, const view& gmat, const view& Emat,
                       view& value_array)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx idx = inner_range.scale_index(lin_i);
        Idx itr = static_cast<Idx>(lin_i);

        Idx i   = itr / nghost;
        Idx i_g = itr % nghost;
        Idx i_arr = (i_g * side_length + i) * 7;

        // Marshalling all main variables at `idx` into contiguous blocks in `value_array`
        value_array[i_arr+0] =  rho[idx];
        value_array[i_arr+1] = umat[idx];
        value_array[i_arr+2] = vmat[idx];
        value_array[i_arr+3] = pmat[idx];
        value_array[i_arr+4] = cmat[idx];
        value_array[i_arr+5] = gmat[idx];
        value_array[i_arr+6] = Emat[idx];
    });
}


extern "C"
void write_border_array(const Range& range, const InnerRange1D& inner_range,
                        Idx nghost, Idx side_length,
                        view& rho, view& umat, view& vmat, view& pmat,
                        view& cmat, view& gmat, view& Emat,
                        const view& value_array)
{
    parallel_kernel(range,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx idx = inner_range.scale_index(lin_i);
        Idx itr = static_cast<Idx>(lin_i);

        Idx i   = itr / nghost;
        Idx i_g = itr % nghost;
        Idx i_arr = (i_g * side_length + i) * 7;

        // Unmarshalling all main variables from contiguous blocks in `value_array` to `idx`
         rho[idx] = value_array[i_arr+0];
        umat[idx] = value_array[i_arr+1];
        vmat[idx] = value_array[i_arr+2];
        pmat[idx] = value_array[i_arr+3];
        cmat[idx] = value_array[i_arr+4];
        gmat[idx] = value_array[i_arr+5];
        Emat[idx] = value_array[i_arr+6];
    });
}
