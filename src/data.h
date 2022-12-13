
#ifndef ARMON_KOKKOS_DATA_H
#define ARMON_KOKKOS_DATA_H

#include <Kokkos_Core.hpp>

#include "parameters.h"


using view = Kokkos::View<flt_t*>;
using host_view = view::HostMirror;


template<typename view_t>
struct DataHolder
{
    view_t x, y;
    view_t rho, umat, vmat, Emat, pmat, cmat, gmat, ustar, pstar;
    view_t work_array_1, work_array_2, work_array_3, work_array_4;
    view_t domain_mask;

    DataHolder() = default;

    [[maybe_unused]]
    DataHolder(const std::string& label, int size);

    [[nodiscard]]
    DataHolder<typename view_t::HostMirror> as_mirror() const;

    template<typename mirror_view_t>
    void deep_copy_to_mirror(DataHolder<mirror_view_t>& mirror) const;
};


using Data = DataHolder<view>;
using HostData = DataHolder<host_view>;


template<typename view_t>
DataHolder<view_t>::DataHolder(const std::string &label, int size)
        : x(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , y(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , rho(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , umat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , vmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , Emat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , pmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , cmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , gmat(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , ustar(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , pstar(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , work_array_1(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , work_array_2(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , work_array_3(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , work_array_4(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
        , domain_mask(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), size)
{ }


template<typename view_t>
DataHolder<typename view_t::HostMirror> DataHolder<view_t>::as_mirror() const {
    DataHolder<typename view_t::HostMirror> mirror;
    mirror.x = Kokkos::create_mirror_view(x);
    mirror.y = Kokkos::create_mirror_view(y);
    mirror.rho = Kokkos::create_mirror_view(rho);
    mirror.umat = Kokkos::create_mirror_view(umat);
    mirror.vmat = Kokkos::create_mirror_view(vmat);
    mirror.Emat = Kokkos::create_mirror_view(Emat);
    mirror.pmat = Kokkos::create_mirror_view(pmat);
    mirror.cmat = Kokkos::create_mirror_view(cmat);
    mirror.gmat = Kokkos::create_mirror_view(gmat);
    mirror.ustar = Kokkos::create_mirror_view(ustar);
    mirror.pstar = Kokkos::create_mirror_view(pstar);
    mirror.work_array_1 = Kokkos::create_mirror_view(work_array_1);
    mirror.work_array_2 = Kokkos::create_mirror_view(work_array_2);
    mirror.work_array_3 = Kokkos::create_mirror_view(work_array_3);
    mirror.work_array_4 = Kokkos::create_mirror_view(work_array_4);
    mirror.domain_mask = Kokkos::create_mirror_view(domain_mask);
    return mirror;
}


template<typename view_t>
template<typename mirror_view_t>
void DataHolder<view_t>::deep_copy_to_mirror(DataHolder<mirror_view_t> &mirror) const {
    Kokkos::deep_copy(mirror.x, x);
    Kokkos::deep_copy(mirror.y, y);
    Kokkos::deep_copy(mirror.rho, rho);
    Kokkos::deep_copy(mirror.umat, umat);
    Kokkos::deep_copy(mirror.vmat, vmat);
    Kokkos::deep_copy(mirror.Emat, Emat);
    Kokkos::deep_copy(mirror.pmat, pmat);
    Kokkos::deep_copy(mirror.cmat, cmat);
    Kokkos::deep_copy(mirror.gmat, gmat);
    Kokkos::deep_copy(mirror.ustar, ustar);
    Kokkos::deep_copy(mirror.pstar, pstar);
    Kokkos::deep_copy(mirror.work_array_1, work_array_1);
    Kokkos::deep_copy(mirror.work_array_2, work_array_2);
    Kokkos::deep_copy(mirror.work_array_3, work_array_3);
    Kokkos::deep_copy(mirror.work_array_4, work_array_4);
    Kokkos::deep_copy(mirror.domain_mask, domain_mask);
}

#endif //ARMON_KOKKOS_DATA_H
