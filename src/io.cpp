
#include "io.h"

#include "indexing.h"

#include <fstream>
#include <iomanip>


void write_output(const Params& p, const HostData& d, const char* file_name)
{
    std::fstream file(file_name, std::ios::out);

    int j_deb =    0 + (p.write_ghosts ? -p.nb_ghosts : 0);
    int j_fin = p.ny + (p.write_ghosts ? +p.nb_ghosts : 0);
    int i_deb =    0 + (p.write_ghosts ? -p.nb_ghosts : 0);
    int i_fin = p.nx + (p.write_ghosts ? +p.nb_ghosts : 0);

    const auto vars = d.main_vars_array();

    file << std::scientific;
    file.precision(p.output_precision);
    const int width = p.output_precision + 3;

    for (int j = j_deb; j < j_fin; j++) {
        for (int i = i_deb; i < i_fin; i++) {
            int idx = index_1D(p, i, j);

            auto it = vars.cbegin();
            file << std::setw(width) << (*it)->operator[](idx);
            for (it++; it != vars.cend(); it++) {
                file << ", " << std::setw(width) << (*it)->operator[](idx);
            }

            file << "\n";
        }

        file << "\n"; // For pm3d display with gnuplot
    }

    if (p.verbose < 2) {
        std::cout << "Wrote to file: " << file_name << "\n";
    }
}


void write_output(const Params& p, const HostData& d)
{
    return write_output(p, d, p.output_file);
}


void load_data(const Params& p, HostData& d, std::istream& file)
{
    int j_deb =    0 + (p.write_ghosts ? -p.nb_ghosts : 0);
    int j_fin = p.ny + (p.write_ghosts ? +p.nb_ghosts : 0);
    int i_deb =    0 + (p.write_ghosts ? -p.nb_ghosts : 0);
    int i_fin = p.nx + (p.write_ghosts ? +p.nb_ghosts : 0);

    const auto vars = d.main_vars_array();

    for (int j = j_deb; j < j_fin; j++) {
        for (int i = i_deb; i < i_fin; i++) {
            int idx = index_1D(p, i, j);

            auto it = vars.cbegin();
            file >> (*it)->operator[](idx);
            for (it++; it != vars.cend(); it++) {
                file.ignore(1, ',');
                file >> (*it)->operator[](idx);
            }
        }
    }
}


bool is_approx(flt_t a, flt_t b, flt_t tol)
{
    constexpr flt_t rtol = 4 * std::numeric_limits<flt_t>::epsilon();
    return std::abs(a - b) <= std::max(tol, rtol * std::max(std::abs(a), std::abs(b)));
}


int compare_with_reference(const Params& params, const HostData& ref_data, const HostData& data)
{
    auto ref_vars = ref_data.main_vars_array();
    auto vars = data.main_vars_array();

    int total_differences_count = 0;
    int comp_count = 0;

    for (int j = 0; j < params.ny; j++) {
        int row_deb = index_1D(params, 0, j);
        int row_fin = index_1D(params, params.nx, j);

        auto ref_it = ref_vars.cbegin(), it = vars.cbegin();
        for (int field_i = 0; field_i < ref_vars.size(); field_i++, ref_it++, it++) {
            int row_diff_count = 0;
            flt_t max_diff = 0;
            for (int idx = row_deb; idx < row_fin; idx++) {
                flt_t ref_val = (**ref_it)(idx);
                flt_t val = (**it)(idx);
                bool is_eq = is_approx(ref_val, val);
                row_diff_count += !is_eq;
                max_diff = std::max(max_diff, std::abs(ref_val - val));

                if (!is_eq) {
                    diff_start = std::min(idx, diff_start);
                    diff_end = std::max(idx, diff_end);
                }
            }

            total_differences_count += row_diff_count;

            if (row_diff_count > 0) {
                std::streamsize tmp_precision = std::cout.precision();
                std::cout.precision(std::numeric_limits<flt_t>::digits10);
                std::cout << "Row " << std::setw(3) << (j+1)
                          << " has " << std::setw(3) << row_diff_count
                          << " differences in '" << std::setw(4) << (**ref_it).label()
                          << "' from " << std::setw(3) << (diff_start - row_deb + 1)
                          << " to " << std::setw(3) << (diff_end - row_deb + 1)
                          << ", max diff: " << max_diff << "\n";
                std::cout.precision(tmp_precision);
            }
        }
    }

    return total_differences_count;
}


bool compare_with_file(const Params& p, const HostData& d, const std::string& ref_file_name)
{
    std::ifstream ref_file(ref_file_name);
    ref_file.exceptions(std::ios::failbit | std::istream::badbit);

    HostData ref_data(p.nb_cells);
    load_data(p, ref_data, ref_file);

    bool is_different = compare_with_reference(p, ref_data, d) > 0;

    return is_different;
}
