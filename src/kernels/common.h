
#ifndef ARMON_KOKKOS_COMMON_H
#define ARMON_KOKKOS_COMMON_H

#include <Kokkos_Core.hpp>


#if USE_SINGLE_PRECISION
using flt_t = float;
#else
using flt_t = double;
#endif


#ifdef __GNUC__
#define DLL_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif


using view = Kokkos::View<flt_t*>;
using host_view = view::HostMirror;
using mask_view = Kokkos::View<flt_t*>;
using host_mask_view = mask_view::HostMirror;


enum class Axis : int {
    X = 0,
    Y = 1
};


enum class Side : int {
    Left,
    Right,
    Top,
    Bottom
};


inline std::string to_string(Axis axis)
{
    switch (axis) {
    default: [[fallthrough]];
    case Axis::X: return "X";
    case Axis::Y: return "Y";
    }
}


inline std::string to_string(Side side)
{
    switch (side) {
    default: [[fallthrough]];
    case Side::Left:   return "Left";
    case Side::Right:  return "Right";
    case Side::Top:    return "Top";
    case Side::Bottom: return "Bottom";
    }
}

#endif //ARMON_KOKKOS_COMMON_H
