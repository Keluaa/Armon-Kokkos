
#include "indexing.h"


void DomainRange::inflate_dir(Axis axis, long n)
{
    switch (axis) {
    case Axis::X:
        row_start -= row_step * n;
        row_end   += row_step * n;
        break;
    case Axis::Y:
        col_start -= col_step * n;
        col_end   += col_step * n;
        break;
    }
}


void DomainRange::expand_dir(Axis axis, long n)
{
    switch (axis) {
    case Axis::X:
        row_end += row_step * n;
        break;
    case Axis::Y:
        col_end += col_step * n;
        break;
    }
}


unsigned long range_length(long start, long step, long stop)
{
    // See `length` in 'base/range.jl:762', simplified for the case where 'step >= 0'
    if ((start != stop) && ((step > 0) != (stop > start))) {
        return 0; // Empty range. See `isempty` in 'base/range.jl:668'
    }
    Idx diff = stop - start;
    return (diff / step) + 1;
}


long range_last(long start, long step, long stop)
{
    // See `steprange_last` in 'base/range.jl:326', simplified for the case where 'step >= 0'
    if (stop == start) {
        return stop;
    } else {
        long remainder = (stop - start) % step;
        return stop - remainder;
    }
}


unsigned long DomainRange::col_length() const
{
    return range_length(col_start, col_step, col_end);
}


unsigned long DomainRange::row_length() const
{
    return range_length(row_start, row_step, row_end);
}


unsigned long DomainRange::length() const
{
    return col_length() * row_length();
}


std::tuple<Range, InnerRange1D> DomainRange::directIter1D() const
{
    return {
            { 0, static_cast<Idx>(length()) },
            { static_cast<Idx>(begin()), static_cast<Idx>(row_step) }
    };
}


std::tuple<Range, InnerRange1D> DomainRange::iter1D() const
{
    return {
            { 0, static_cast<Idx>(end() - begin() + 1) },
            { static_cast<Idx>(begin()), static_cast<Idx>(row_step) }
    };
}


std::tuple<Range, InnerRange2D> DomainRange::iter2D() const
{
    return {
            { 0, static_cast<Idx>(length()) },
            { static_cast<Idx>(col_start), static_cast<Idx>(col_step), static_cast<Idx>(row_start), static_cast<UIdx>(row_length()) }
    };
}


long DomainRange::begin() const
{
    return col_start + row_start;
}


long DomainRange::end() const
{
    return range_last(col_start, col_step, col_end) + range_last(row_start, row_step, row_end);
}


double overwork_factor(long N, long T, long tile_y_size)
{
    // "total CPU time" spent working on the iterations of the domain, excluding the barrier at the end:
    // the (fractional) amount of tiles to work on for the loop
    double useful_work_area = double(N) / double(tile_y_size);

    // "total CPU time" of the iteration of the domain, including the barrier at the end:
    // if all threads worked at 100% during the whole loop, this would be the number of tiles that could be processed
    double total_time_area = std::ceil(useful_work_area / double(T)) * double(T);

    double of = 1 - (useful_work_area / total_time_area);

//    printf("%ld tiles of (%ld x M) spread on %ld threads: \n"
//           " - useful work area: %g, total work area: %g\n"
//           " - this is %s at %g\n",
//           N / tile_y_size, tile_y_size, T, useful_work_area, total_time_area, of);
    return of;
}


static std::map<std::pair<long, long>, long> optimal_iter_space_cache;
static std::mutex optimal_iter_space_cache_mutex;


long optimal_split_iter_space(long N, long threads, double overwork_tolerance)
{
    {
        // Cache the results: while the algorithm is `O(threads)`, it must be called before every kernel, adding much
        // unnecessary overhead.
        std::lock_guard<std::mutex> lk(optimal_iter_space_cache_mutex);
        auto optimal_iter_space = optimal_iter_space_cache.find({N, threads});
        if (optimal_iter_space != optimal_iter_space_cache.end()) {
            return optimal_iter_space->second;
        }
    }

    long min_tile_size = std::min(4L, threads);

    // To limit fragmentation, we want to maximize the size of tiles, while keeping the amount of overwork minimal.
    // We keep the biggest tile size with an overwork factor below the tolerance, or, if none have an acceptable amount
    // of overwork, the tile size with the minimum of overwork.
    double min_overwork_factor = INFINITY;
    long optimal_tile_y_size = 0;
    double biggest_acceptable_tile_y_size_overwork_factor = 0;
    long biggest_acceptable_tile_y_size = 0;
    for (long tile_y_size = threads; tile_y_size > 0; tile_y_size--) {
        if (tile_y_size < min_tile_size) continue;
        double of = overwork_factor(N, threads, tile_y_size);
        if (of == 0) {
            // No overwork: optimal tiling, with the biggest possible tiles (this is why we iterate from T to 1)
            biggest_acceptable_tile_y_size = tile_y_size;
            break;
        } else if (of < overwork_tolerance &&
                (biggest_acceptable_tile_y_size < tile_y_size || 2*of < biggest_acceptable_tile_y_size_overwork_factor)) {
            // Keep tile sizes with a x2 improvement
            biggest_acceptable_tile_y_size = tile_y_size;
            biggest_acceptable_tile_y_size_overwork_factor = of;
        } else if (of < min_overwork_factor) {
            min_overwork_factor = of;
            optimal_tile_y_size = tile_y_size;
        }
    }

    if (biggest_acceptable_tile_y_size > 0) {
        optimal_tile_y_size = biggest_acceptable_tile_y_size;
    }

    if (optimal_tile_y_size == 0) {
        throw std::logic_error("Cannot choose an optimal tile size for "
            + std::to_string(N) + " iterations on " + std::to_string(threads) + " threads");
    }

    {
        std::lock_guard<std::mutex> lk(optimal_iter_space_cache_mutex);
        optimal_iter_space_cache[{N, threads}] = optimal_tile_y_size;
    }

    return optimal_tile_y_size;
}
