
#ifndef ARMON_KOKKOS_DATA_H
#define ARMON_KOKKOS_DATA_H

#include <Kokkos_Core.hpp>

#include "parameters.h"


using view = Kokkos::View<flt_t*>;
using host_view = view::HostMirror;
using mask_view = Kokkos::View<bool*>;
using host_mask_view = mask_view::HostMirror;


template<typename view_t, typename mask_view_t>
struct DataHolder
{
    view_t x, y;
    view_t rho, umat, vmat, Emat, pmat, cmat, gmat, ustar, pstar;
    view_t work_array_1, work_array_2, work_array_3, work_array_4;
    mask_view_t domain_mask;

    DataHolder() = default;

    [[maybe_unused]]
    explicit DataHolder(int size, const std::string& label = "");

    [[nodiscard]]
    DataHolder<typename view_t::HostMirror, typename mask_view_t::HostMirror> as_mirror() const;

    template<typename mirror_view_t, typename mirror_mask_view_t>
    void deep_copy_to_mirror(DataHolder<mirror_view_t, mirror_mask_view_t>& mirror) const;

    [[nodiscard]] auto vars_array() const;
    [[nodiscard]] auto vars_array();

    [[nodiscard]] auto main_vars_array() const;
    [[nodiscard]] auto main_vars_array();
};


using Data = DataHolder<view, mask_view>;
using HostData = DataHolder<host_view, host_mask_view>;


template<typename view_t, typename mask_view_t>
DataHolder<view_t, mask_view_t>::DataHolder(int size, const std::string& label)
        : x(Kokkos::view_alloc(label + "x", Kokkos::WithoutInitializing), size)
        , y(Kokkos::view_alloc(label + "y", Kokkos::WithoutInitializing), size)
        , rho(Kokkos::view_alloc(label + "rho", Kokkos::WithoutInitializing), size)
        , umat(Kokkos::view_alloc(label + "umat", Kokkos::WithoutInitializing), size)
        , vmat(Kokkos::view_alloc(label + "vmat", Kokkos::WithoutInitializing), size)
        , Emat(Kokkos::view_alloc(label + "Emat", Kokkos::WithoutInitializing), size)
        , pmat(Kokkos::view_alloc(label + "pmat", Kokkos::WithoutInitializing), size)
        , cmat(Kokkos::view_alloc(label + "cmat", Kokkos::WithoutInitializing), size)
        , gmat(Kokkos::view_alloc(label + "gmat", Kokkos::WithoutInitializing), size)
        , ustar(Kokkos::view_alloc(label + "ustar", Kokkos::WithoutInitializing), size)
        , pstar(Kokkos::view_alloc(label + "pstar", Kokkos::WithoutInitializing), size)
        , work_array_1(Kokkos::view_alloc(label + "work_array_1", Kokkos::WithoutInitializing), size)
        , work_array_2(Kokkos::view_alloc(label + "work_array_2", Kokkos::WithoutInitializing), size)
        , work_array_3(Kokkos::view_alloc(label + "work_array_3", Kokkos::WithoutInitializing), size)
        , work_array_4(Kokkos::view_alloc(label + "work_array_4", Kokkos::WithoutInitializing), size)
        , domain_mask(Kokkos::view_alloc(label + "domain_mask", Kokkos::WithoutInitializing), size)
{ }


template<typename view_t, typename mask_view_t>
DataHolder<typename view_t::HostMirror, typename mask_view_t::HostMirror>
DataHolder<view_t, mask_view_t>::as_mirror() const
{
    DataHolder<typename view_t::HostMirror, typename mask_view_t::HostMirror> mirror;

    auto our_vars = vars_array();
    auto mirror_vars = mirror.vars_array();
    auto our_it = our_vars.cbegin();
    auto mirror_it = mirror_vars.begin();
    for (; our_it != our_vars.cend() && mirror_it != mirror_vars.cend(); our_it++, mirror_it++) {
        **mirror_it = Kokkos::create_mirror_view(**our_it);
    }
    mirror.domain_mask = Kokkos::create_mirror_view(domain_mask);
    return mirror;
}


template<typename view_t, typename mask_view_t>
template<typename mirror_view_t, typename mirror_mask_view_t>
void DataHolder<view_t, mask_view_t>::deep_copy_to_mirror(DataHolder<mirror_view_t, mirror_mask_view_t>& mirror) const
{
    auto our_vars = vars_array();
    auto mirror_vars = mirror.vars_array();
    auto our_it = our_vars.cbegin();
    auto mirror_it = mirror_vars.begin();
    for (; our_it != our_vars.cend() && mirror_it != mirror_vars.cend(); our_it++, mirror_it++) {
        Kokkos::deep_copy(**mirror_it, **our_it);
    }
    Kokkos::deep_copy(mirror.domain_mask, domain_mask);
}


template<typename view_t, typename mask_view_t>
auto DataHolder<view_t, mask_view_t>::vars_array()
{
    return std::array{
        &x, &y, &rho, &umat, &vmat, &Emat, &pmat, &cmat, &gmat, &ustar, &pstar,
        &work_array_1, &work_array_2, &work_array_3, &work_array_4
    };
}

template<typename view_t, typename mask_view_t>
auto DataHolder<view_t, mask_view_t>::vars_array() const
{
    return std::array{
        &x, &y, &rho, &umat, &vmat, &Emat, &pmat, &cmat, &gmat, &ustar, &pstar,
        &work_array_1, &work_array_2, &work_array_3, &work_array_4
    };
}


template<typename view_t, typename mask_view_t>
auto DataHolder<view_t, mask_view_t>::main_vars_array()
{
    return { &x, &y, &rho, &umat, &vmat, &pmat };
}


template<typename view_t, typename mask_view_t>
auto DataHolder<view_t, mask_view_t>::main_vars_array() const
{
    return { &x, &y, &rho, &umat, &vmat, &pmat };
}

#endif //ARMON_KOKKOS_DATA_H
