
#ifndef ARMON_KOKKOS_INDEXING_H
#define ARMON_KOKKOS_INDEXING_H

#include <Kokkos_Core.hpp>

#include "common.h"


/**
 * The default index type: the first argument to kernels.
 * Usually an `unsigned long` or `unsigned int`, it can be an `long` or an `int` in some backends.
 */
using UIdx = Kokkos::RangePolicy<>::index_type;

/**
 * Signed version of `UIdx`
 */
using Idx = std::make_signed_t<UIdx>;

using Index_t = Kokkos::IndexType<UIdx>;
using Team_t = Kokkos::TeamPolicy<Index_t>::member_type;


/**
 * An open interval: `[start, end)`. This is the linear index range visible to Kokkos.
 */
struct Range {
    Idx start;
    Idx end;  // exclusive (open interval)

    [[nodiscard]] Idx length() const { return end - start; }
};


/**
 * Used to scale the linear index of in a kernel into a valid index in a 1D array (or in an 1D iteration).
 */
struct InnerRange1D {
    Idx start;
    Idx step;

    [[nodiscard]] KOKKOS_INLINE_FUNCTION
    Idx scale_index(UIdx i) const { return static_cast<Idx>(i) * step + start; }
};


/**
 * Used to scale the linear index of in a kernel into a valid index in a 2D array (or in an 2D iteration).
 *
 * It represents an iteration in a 2D domain of `Nx × Ny`, with `Ng` ghost cells padding all 4 sides.
 * Rows are always iterated along the X dimension.
 *  - the Y axis ("main" axis) is iterated as the range: `main_range_start:main_range_step:main_range_end`, which gives
 *    the first index of the row. `main_range_end` is implicitly given to Kokkos in the main `Range`.
 *  - the X axis starts the first index given Y axis range: `row_range_start:1:row_range_end` with
 *    `row_range_start - row_range_end == row_range_length`. The stride is always 1.
 *
 * It is based on its Julia equivalent: the notion of ranges are implemented in a similar way.
 */
struct InnerRange2D {
    Idx main_range_start;
    Idx main_range_step;
    Idx row_range_start;
    UIdx row_range_length;

    [[nodiscard]] KOKKOS_INLINE_FUNCTION
    Idx scale_index(UIdx i) const {
        Idx ix = static_cast<Idx>(i / row_range_length);
        Idx iy = static_cast<Idx>(i % row_range_length);
        Idx j = main_range_start + ix * main_range_step;
        return row_range_start + iy + j;
    }
};


/**
 * Same as `InnerRange2D` but in a `Kokkos::parallel_for` with a `Kokkos::MDRangePolicy`, allowing to move most of the
 * indexing logic in the loops, at the cost of having 2 indexing variables instead of 1.
 */
struct MDInnerRange2D {
    Idx main_range_step;

    [[nodiscard]] KOKKOS_INLINE_FUNCTION
    Idx scale_index(UIdx j, UIdx i) const {
        return j * main_range_step + i;
    }
};


/**
 * Represents a 2D iteration on a 2D array, using two ranges, one along the columns, another along the rows.
 */
struct DomainRange {
    // `col_end` and `row_end` are inclusive
    long col_start, col_step, col_end;
    long row_start, row_step, row_end;

    void inflate_dir(Axis axis, long n);
    void expand_dir(Axis axis, long n);

    [[nodiscard]] unsigned long col_length() const;
    [[nodiscard]] unsigned long row_length() const;
    [[nodiscard]] unsigned long length() const;

    [[nodiscard]] long begin() const;
    [[nodiscard]] long end() const;  // inclusive

    [[nodiscard]] std::tuple<Range, InnerRange1D> directIter1D() const;
    [[nodiscard]] std::tuple<Range, InnerRange1D> iter1D() const;
    [[nodiscard]] std::tuple<Range, InnerRange2D> iter2D() const;

    bool operator==(const DomainRange& other) const {
        return col_start == other.col_start && col_step == other.col_step && col_end == other.col_end
            && row_start == other.row_start && row_step == other.row_step && row_end == other.row_end;
    }
};


/**
 * Iterate `range` using a simple `RangePolicy`.
 */
inline Kokkos::RangePolicy<Index_t> iter_lin(const Range& range)
{
    return { static_cast<UIdx>(range.start), static_cast<UIdx>(range.end) };
}


/**
 * Iterate `range` using hierarchical parallelism with `TeamPolicy`.
 *
 * The number of threads per team is automatic.
 */
inline Kokkos::TeamPolicy<Index_t> iter_simd(const Range& range, int V)
{
    int size = static_cast<int>(Kokkos::ceil(static_cast<double>(range.length()) / V));
    return { size, Kokkos::AUTO, V };
}


inline std::tuple<std::tuple<Kokkos::TeamPolicy<Index_t>, std::tuple<long, long, long>>, MDInnerRange2D>
        iter_2d(const Range& range, const InnerRange2D& inner_range)
{
    long first_i = inner_range.scale_index(range.start);
    long last_i  = inner_range.scale_index(range.end - 1);

    long start_x = first_i % inner_range.main_range_step;
    long start_y = first_i / inner_range.main_range_step;
    long end_x   = last_i  % inner_range.main_range_step + 1;
    long end_y   = last_i  / inner_range.main_range_step + 1;

    return {
            { Kokkos::TeamPolicy<Index_t>(end_y - start_y, Kokkos::AUTO), std::make_tuple(start_y, start_x, end_x) },
            { inner_range.main_range_step }
    };
}


double overwork_factor(long N, long T, long tile_y_size);
long optimal_split_iter_space(long N, long threads, double overwork_tolerance = 0.10);


inline std::tuple<Kokkos::MDRangePolicy<Kokkos::Rank<2>, Index_t>, MDInnerRange2D>
        iter_md(const Range& range, const InnerRange2D& inner_range)
{
    long first_i = inner_range.scale_index(range.start);
    long last_i  = inner_range.scale_index(range.end - 1);

    long start_x = first_i % inner_range.main_range_step;
    long start_y = first_i / inner_range.main_range_step;
    long end_x   = last_i  % inner_range.main_range_step + 1;
    long end_y   = last_i  / inner_range.main_range_step + 1;

#ifdef KOKKOS_ENABLE_OPENMP
    // Very important for NUMA => group tiles by threads.
    // The default is a round-robin distribution amongst threads, which is terrible.
    // A size of `0` means Kokkos default.
    long max_threads = Kokkos::OpenMP::concurrency();
    long tile_y_size = std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP> && BALANCE_MD_ITER
            ? optimal_split_iter_space(end_y - start_y, max_threads)
            : 0;
#else
    long tile_y_size = 0;
#endif

    return {
        { { start_y, start_x }, { end_y, end_x }, { tile_y_size, 0 } },
        { inner_range.main_range_step }
    };
}


inline auto iter(const Range& range, const InnerRange2D& inner_range)
{
#if USE_2D_ITER
    return iter_2d(range, inner_range);
#elif USE_MD_ITER
    return iter_md(range, inner_range);
#else
    return std::make_tuple(range, inner_range);
#endif  // USE_MD_ITER
}


#if USE_2D_ITER || USE_MD_ITER
#define ITER_IDX_DEF const UIdx _ij, const UIdx _ix
#define ITER_IDX _ij, _ix
#else
#define ITER_IDX_DEF const UIdx _lin_i
#define ITER_IDX _lin_i
#endif  // USE_MD_ITER

#endif // ARMON_KOKKOS_INDEXING_H
