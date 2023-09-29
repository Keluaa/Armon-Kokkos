
#ifndef ARMON_KOKKOS_DOUBLE_REDUCER_H
#define ARMON_KOKKOS_DOUBLE_REDUCER_H

#include <type_traits>
#include <tuple>

#include <Kokkos_Core.hpp>


/**
 * Allows to reduce two values at the same time in a Kokkos `parallel_reduce` kernel.
 *
 * While `parallel_reduce` already has the semantics to do so, variadic templating is very hard (and annoying), making
 * supporting this syntax difficult. Furthermore, this syntax is currently not supported by all backends.
 *
 * Using this Reducer class shouldn't introduce any overhead (hopefully).
 */
template<typename Reducer>
class DoubleReducer
{
public:
    using reducer = DoubleReducer<Reducer>;
    using value_type = std::tuple<typename Reducer::value_type, typename Reducer::value_type>;
    using result_view_type = Kokkos::View<value_type, typename Reducer::result_view_type::memory_space>;

private:
    const Reducer delegate;
    result_view_type value;

public:

    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const {
        delegate.join(std::get<0>(dest), std::get<0>(src));
        delegate.join(std::get<1>(dest), std::get<1>(src));
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        delegate.init(std::get<0>(val));
        delegate.init(std::get<1>(val));
    }

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const { return *value.data(); }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return value; }

    KOKKOS_INLINE_FUNCTION
    DoubleReducer()
        : delegate()
        , value()
    { }

    KOKKOS_INLINE_FUNCTION
    DoubleReducer(Reducer&& r1, Reducer&& r2)
        : delegate(r1)
        , value()
    {
        value() = std::make_tuple(r1.reference(), r2.reference());
    }

    KOKKOS_INLINE_FUNCTION
    DoubleReducer(value_type& value)
        : delegate(std::get<0>(value))  // Dummy constructor call
        , value(&value)
    { }

    KOKKOS_INLINE_FUNCTION
    DoubleReducer(const result_view_type& value)
        : delegate(std::get<0>(value()))  // Dummy constructor call
        , value(value)
    { }
};


template <typename T>
struct Kokkos::reduction_identity<std::tuple<T, T>> {
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> sum() {
        return std::make_tuple(Kokkos::reduction_identity<T>::sum(), Kokkos::reduction_identity<T>::sum());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> prod() {
        return std::make_tuple(Kokkos::reduction_identity<T>::prod(), Kokkos::reduction_identity<T>::prod());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> max() {
        return std::make_tuple(Kokkos::reduction_identity<T>::max(), Kokkos::reduction_identity<T>::max());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> min() {
        return std::make_tuple(Kokkos::reduction_identity<T>::min(), Kokkos::reduction_identity<T>::min());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> bor() {
        return std::make_tuple(Kokkos::reduction_identity<T>::bor(), Kokkos::reduction_identity<T>::bor());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> band() {
        return std::make_tuple(Kokkos::reduction_identity<T>::band(), Kokkos::reduction_identity<T>::band());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> lor() {
        return std::make_tuple(Kokkos::reduction_identity<T>::lor(), Kokkos::reduction_identity<T>::lor());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static std::tuple<T, T> land() {
        return std::make_tuple(Kokkos::reduction_identity<T>::land(), Kokkos::reduction_identity<T>::land());
    }
};


#ifdef KOKKOS_ENABLE_OPENMPTARGET

template <class Reducer>
struct Kokkos::Impl::OpenMPTargetReducerWrapper<DoubleReducer<Reducer>> {
public:
    // Required
    using value_type = std::remove_cv_t<typename DoubleReducer<Reducer>::value_type>;

    // Required
    KOKKOS_INLINE_FUNCTION
    static void join(value_type& dest, const value_type& src) {
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(std::get<0>(dest), std::get<0>(src));
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(std::get<1>(dest), std::get<1>(src));
    }

    KOKKOS_INLINE_FUNCTION
    static void join(volatile value_type& dest, const volatile value_type& src) {
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(std::get<0>(dest), std::get<0>(src));
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(std::get<1>(dest), std::get<1>(src));
    }

    KOKKOS_INLINE_FUNCTION
    static void init(value_type& val) {
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().init(std::get<0>(val));
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().init(std::get<1>(val));
    }
};

#endif


#endif //ARMON_KOKKOS_DOUBLE_REDUCER_H
