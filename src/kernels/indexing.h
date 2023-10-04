
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
KOKKOS_INLINE_FUNCTION Kokkos::RangePolicy<Index_t> iter(const Range&& range)
{
    return { static_cast<UIdx>(range.start), static_cast<UIdx>(range.end) };
}


KOKKOS_INLINE_FUNCTION Kokkos::RangePolicy<Index_t> iter(const Range& range)
{
    return iter(std::forward<const Range>(range));
}


/**
 * Iterate `range` using hierarchical parallelism with `TeamPolicy`.
 *
 * The number of threads per team is automatic.
 */
KOKKOS_INLINE_FUNCTION Kokkos::TeamPolicy<Index_t> iter_simd(const Range&& range, int V)
{
    int size = static_cast<int>(Kokkos::ceil(static_cast<double>(range.length()) / V));
    return { size, Kokkos::AUTO, V };
}


KOKKOS_INLINE_FUNCTION Kokkos::TeamPolicy<Index_t> iter_simd(const Range& range, int V)
{
    return iter_simd(std::forward<const Range>(range), V);
}


#endif //ARMON_KOKKOS_INDEXING_H
