
#ifndef ARMON_KOKKOS_PARALLEL_KERNELS
#define ARMON_KOKKOS_PARALLEL_KERNELS

#include <Kokkos_Core.hpp>

#include "indexing.h"


#if KOKKOS_VERSION >= 40000
#include <Kokkos_SIMD.hpp>

template<typename T>
constexpr unsigned int default_vector_size()
{
    return Kokkos::Experimental::native_simd<T>::size();
}
#else
template<typename T>
constexpr unsigned int default_vector_size()
{
    return 8;  // Should be OK for most CPU and GPU backends
}
#endif // KOKKOS_VERSION >= 40000


/**
 * The maximum SIMD vector size for the type `T` of the `ExecSpace`.
 */
template<typename ExecSpace, typename T>
constexpr unsigned int get_vector_size()
{
#ifdef KOKKOS_ENABLE_OPENMP
    // Kokkos is unable to correctly detect SIMD register sizes for some CPU: we do it ourselves.
    return (std::is_same_v<ExecSpace, Kokkos::OpenMP>) ? PREFERRED_SIMD_SIZE / sizeof(T) : default_vector_size<T>();
#elif defined(KOKKOS_ENABLE_SERIAL)
    return (std::is_same_v<ExecSpace, Kokkos::Serial>) ? PREFERRED_SIMD_SIZE / sizeof(T) : default_vector_size<T>();
#else
    return default_vector_size<T>();
#endif
}


template<typename Functor>
void parallel_kernel(const Range& range, const Functor& functor)
{
#if USE_SIMD_KERNELS
    constexpr unsigned int V = get_vector_size<Kokkos::DefaultExecutionSpace, flt_t>();
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
    constexpr unsigned int V = get_vector_size<Kokkos::DefaultExecutionSpace, flt_t>();

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

#endif // ARMON_KOKKOS_PARALLEL_KERNELS
