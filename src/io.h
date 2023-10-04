
#ifndef ARMON_KOKKOS_IO_H
#define ARMON_KOKKOS_IO_H

#include "parameters.h"
#include "data.h"

#include <istream>

void write_output(const Params& p, const HostData& d, const char* file_name);
void write_output(const Params& p, const HostData& d);
void load_data(const Params& p, HostData& d, std::istream& file);

bool is_approx(flt_t a, flt_t b, flt_t tol = 1e-13);
int compare_with_reference(const Params& params, const HostData& ref_data, const HostData& data);
bool compare_with_file(const Params& p, const HostData& d, const std::string& ref_file_name);
bool compare_time_step(const Params& p, const std::string& ref_file_name);

#endif //ARMON_KOKKOS_IO_H
