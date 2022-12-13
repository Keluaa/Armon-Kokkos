
#include "kernel_info.h"

#include "indexing.h"

#include <Kokkos_Core.hpp>

#include <tuple>


using dim3d = std::array<unsigned int, 3>;


#ifdef KOKKOS_ENABLE_CUDA

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size(const Policy& policy)
{
    // See Kokkos_Cuda_Parallel.hpp : ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::Cuda>::execute()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i){};
    using FunctorType = decltype(functor);
    using LaunchBounds = typename Policy::launch_bounds;
    using ParallelFor = Kokkos::Impl::ParallelFor<FunctorType, Policy>;

    cudaFuncAttributes attr = Kokkos::Impl::CudaParallelLaunch<ParallelFor, LaunchBounds>::get_cuda_func_attributes();
    const unsigned block_size = Kokkos::Impl::cuda_get_opt_block_size<typename ParallelFor::functor_type, LaunchBounds>(policy.space().impl_internal_space_instance(), attr, functor, 1, 0, 0);

    auto max_grid = Kokkos::Impl::CudaInternal::singleton().m_maxBlock;

    dim3d block{1, block_size, 1};
    dim3d grid{
           std::min(
                   typename Policy::index_type((nb_work + block[1] - 1) / block[1]),
                   typename Policy::index_type(max_grid[0])),
           1, 1};

    return std::make_tuple(block, grid);
}

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size_reduction(const Policy& policy)
{
    // See Kokkos_Cuda_Parallel.hpp : ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType, Kokkos::Cuda>::local_block_size()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i, flt_t& reducer){};
    using FunctorType = decltype(functor);
    using ParallelReduce = Kokkos::Impl::ParallelReduce<FunctorType, Policy, flt_t, Kokkos::Cuda>;
    using return_value_adapter = Kokkos::Impl::ParallelReduceReturnValue<void, flt_t, FunctorType>;

    flt_t return_value{};
    Kokkos::Impl::ParallelReduce<FunctorType, Policy, typename return_value_adapter::reducer_type>
            reduction(functor, policy, return_value_adapter::return_value(return_value, functor));
    const unsigned block_size = reduction.local_block_size(functor);

    dim3d block{1, block_size, 1};
    dim3d grid{std::min(block[1], nb_work), 1, 1};

    return std::make_tuple(block, grid);
}

#else
#ifdef KOKKOS_ENABLE_HIP

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size(const Policy& policy)
{
    // See Kokkos_HIP_Parallel_Range.hpp : ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::Experimental::HIP>::execute()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i){};
    using FunctorType = decltype(functor);
    using LaunchBounds = typename Policy::launch_bounds;
    using ParallelFor = Kokkos::Impl::ParallelFor<FunctorType, Policy>;

    const unsigned block_size = Kokkos::Experimental::Impl::hip_get_preferred_blocksize<ParallelFor, LaunchBounds>();
    const dim3d block{1, block_size, 1};
    const dim3d grid{typename Policy::index_type((nb_work + block[1] - 1) / block[1]), 1, 1};

    return std::make_tuple(block, grid);
}

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size_reduction(const Policy& policy)
{
    // See Kokkos_HIP_Parallel_Range.hpp : ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType, Kokkos::Experimental::HIP>::local_block_size()
    const typename Policy::index_type nb_work = policy.end() - policy.begin();

    auto functor = KOKKOS_LAMBDA(const int i, flt_t& reducer){};
    using FunctorType = decltype(functor);
    using ParallelReduce = Kokkos::Impl::ParallelReduce<FunctorType, Policy, flt_t, Kokkos::Experimental::HIP>;
    using return_value_adapter = Kokkos::Impl::ParallelReduceReturnValue<void, flt_t, FunctorType>;

    flt_t return_value{};
    Kokkos::Impl::ParallelReduce<FunctorType, Policy, typename return_value_adapter::reducer_type>
            reduction(functor, policy, return_value_adapter::return_value(return_value, functor));
    const unsigned block_size = reduction.local_block_size(functor);

    dim3d block{1, block_size, 1};
    dim3d grid{std::min(block[1], typename Policy::index_type((nb_work + block[1] - 1) / block[1])), 1, 1};

    return std::make_tuple(block, grid);
}

#else

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size(const Policy&)
{ return std::make_tuple(dim3d{1, 1, 1}, dim3d{1, 1, 1}); }

template<typename Policy>
std::tuple<dim3d, dim3d> get_block_and_grid_size_reduction(const Policy&)
{ return std::make_tuple(dim3d{1, 1, 1}, dim3d{1, 1, 1}); }

#endif // KOKKOS_ENABLE_HIP
#endif // KOKKOS_ENABLE_CUDA


void print_kernel_params(const Params& p)
{
    auto range = real_domain(p);
    auto [block, grid] = get_block_and_grid_size(iter(range));
    printf("Kernel launch parameters for 'parallel_for', with range [%d, %d]:\n", std::get<0>(range), std::get<1>(range));
    printf(" - block dim: %d, %d, %d\n", block[0], block[1], block[2]);
    printf(" - grid dim:  %d, %d, %d\n", grid[0], grid[1], grid[2]);

    auto [block_r, grid_r] = get_block_and_grid_size_reduction(iter(range));
    printf("Kernel launch parameters for 'parallel_reduce', with range [%d, %d]:\n", std::get<0>(range), std::get<1>(range));
    printf(" - block dim: %d, %d, %d\n", block_r[0], block_r[1], block_r[2]);
    printf(" - grid dim:  %d, %d, %d\n", grid_r[0], grid_r[1], grid_r[2]);
}
