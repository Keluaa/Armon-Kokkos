
#ifndef ARMON_KOKKOS_REFERENCE_H
#define ARMON_KOKKOS_REFERENCE_H

#include "parameters.h"
#include "data.h"

#include <string>

std::string get_reference_data_path(Test test_case);

Params get_reference_params(Test test_case);

std::tuple<HostData, flt_t, int> load_reference_data(const Params& ref_params, const std::string& ref_file_path);

bool is_approx(flt_t a, flt_t b);

int compare_with_reference(const Params& ref_params, const HostData& ref_data, const HostData& data);

#endif //ARMON_KOKKOS_REFERENCE_H
