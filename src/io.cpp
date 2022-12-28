
#include "io.h"

#include "indexing.h"

#include <fstream>


void write_output(const Params& p, const HostData& d, const char* file_name)
{
    FILE* file = fopen(file_name, "w");

    int j_deb =    0 + (p.write_ghosts ? -p.nb_ghosts : 0);
    int j_fin = p.ny + (p.write_ghosts ? +p.nb_ghosts : 0);
    int i_deb =    0 + (p.write_ghosts ? -p.nb_ghosts : 0);
    int i_fin = p.nx + (p.write_ghosts ? +p.nb_ghosts : 0);

    const std::array vars = {&d.x, &d.y, &d.rho, &d.umat, &d.vmat, &d.pmat};

    for (int j = j_deb; j < j_fin; j++) {
        for (int i = i_deb; i < i_fin; i++) {
            int idx = index_1D(p, i, j);

            auto it = vars.cbegin();
            fprintf(file, "%12.9f", (*it)->operator[](idx));
            for (it++; it != vars.cend(); it++) {
                fprintf(file, ", %12.9f", (*it)->operator[](idx));
            }

            fprintf(file, "\n");
        }
    }

    fclose(file);

    if (p.verbose < 2) {
        printf("Wrote to file: %s\n", p.output_file);
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

    const std::array vars = {&d.x, &d.y, &d.rho, &d.umat, &d.vmat, &d.pmat};

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


bool is_approx(flt_t a, flt_t b)
{
    return std::abs(a - b) <= flt_t(1e-13);
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
                row_diff_count += !is_approx(ref_val, val);
                comp_count++;
                max_diff = std::max(max_diff, std::abs(ref_val - val));
            }

            total_differences_count += row_diff_count;

            if (row_diff_count > 0) {
                std::streamsize tmp_precision = std::cout.precision();
                std::cout.precision(std::numeric_limits<flt_t>::digits10);
                std::cout << "Row " << (j+1) << " has " << row_diff_count
                          << " differences (max diff: " << max_diff << ")"
                          << " in '" << (**ref_it).label()
                          << "' with the reference\n";
                std::cout.precision(tmp_precision);
            }
        }
    }

    std::cout << "comp " << comp_count << ", diff: " << total_differences_count << "\n";

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
