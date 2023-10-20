
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
 *
 * Usage:
 * ```
 *     T a, b;
 *     auto value_tuple = Kokkos::make_pair(a, b);
 *     auto reducer = DoubleReducer<Kokkos::Min<T>>(value_tuple);
 *     Kokkos::parallel_reduce(policy, functor, reducer);
 *     std::tie(a, b) = value_tuple.to_std_pair();
 * ```
 */
template<typename Reducer>
class DoubleReducer
{
public:
    using reducer = DoubleReducer<Reducer>;
    using value_type = Kokkos::pair<typename Reducer::value_type, typename Reducer::value_type>;
    using result_view_type = Kokkos::View<value_type, typename Reducer::result_view_type::memory_space>;

private:
    const Reducer delegate;
    result_view_type value;

public:

    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const {
        delegate.join(dest.first, src.first);
        delegate.join(dest.second, src.second);
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        delegate.init(val.first);
        delegate.init(val.second);
    }

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const { return *value.data(); }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return value; }

    KOKKOS_INLINE_FUNCTION
    DoubleReducer(value_type& value)
        : delegate(value.first)  // Dummy constructor call
        , value(&value)
    { }

    KOKKOS_INLINE_FUNCTION
    DoubleReducer(const result_view_type& value)
        : delegate(value.first)  // Dummy constructor call
        , value(value)
    { }
};


template <typename T>
struct Kokkos::reduction_identity<Kokkos::pair<T, T>> {
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> sum() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::sum(), Kokkos::reduction_identity<T>::sum());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> prod() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::prod(), Kokkos::reduction_identity<T>::prod());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> max() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::max(), Kokkos::reduction_identity<T>::max());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> min() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::min(), Kokkos::reduction_identity<T>::min());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> bor() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::bor(), Kokkos::reduction_identity<T>::bor());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> band() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::band(), Kokkos::reduction_identity<T>::band());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> lor() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::lor(), Kokkos::reduction_identity<T>::lor());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static Kokkos::pair<T, T> land() {
        return Kokkos::make_pair(Kokkos::reduction_identity<T>::land(), Kokkos::reduction_identity<T>::land());
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
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(dest.first, src.first);
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(dest.second, src.second);
    }

    KOKKOS_INLINE_FUNCTION
    static void join(volatile value_type& dest, const volatile value_type& src) {
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(dest.first, src.first);
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().join(dest.second, src.second);
    }

    KOKKOS_INLINE_FUNCTION
    static void init(value_type& val) {
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().init(val.first);
        Kokkos::Impl::OpenMPTargetReducerWrapper<Reducer>().init(val.second);
    }
};

#endif


#endif //ARMON_KOKKOS_DOUBLE_REDUCER_H
