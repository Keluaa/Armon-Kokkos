
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


std::tuple<Range, InnerRange1D> DomainRange::iter1D() const
{
    return {
            { 0, static_cast<Idx>(length()) },
            { begin(), row_step }
    };
}


std::tuple<Range, InnerRange2D> DomainRange::iter2D() const
{
    return {
            { 0, static_cast<Idx>(length()) },
            { col_start, col_step, row_start, row_length() }
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
