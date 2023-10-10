
#include "utils.h"

#include "common.h"
#include "indexing.h"


__attribute__((noreturn))
void default_exception_handler(const char* kernel, const char* msg)
{
    printf("Exception in kernel %s: %s\n", kernel, msg);
    exit(1);
}


extern "C" DLL_EXPORT
void (*raise_exception_handler)(const char* kernel, const char* msg) __attribute__((noreturn)) = default_exception_handler;


extern "C" DLL_EXPORT
int idx_size() { return sizeof(Idx); }

extern "C" DLL_EXPORT
bool is_uidx_signed() { return std::is_signed_v<UIdx>; }

extern "C" DLL_EXPORT
int flt_size() { return sizeof(flt_t); }


#ifdef TRY_ALL_CALLS

#if defined(__GNUC__)
#include <cxxabi.h>

const char* current_exception_typename(const std::exception&)
{
    return abi::__cxa_demangle(abi::__cxa_current_exception_type()->name(), nullptr, nullptr, nullptr);
}
#else
#include <typeinfo>

const char* current_exception_typename(const std::exception& exception)
{
    return typeid(exception).name();
}
#endif


__attribute__((noreturn)) void raise_exception(const char* kernel_name, const std::exception& exception)
{
    const char* exception_type = current_exception_typename(exception);
    auto str = "(" + std::string(exception_type) + ") " + exception.what();
    raise_exception_handler(kernel_name, str.c_str());
}

#endif //TRY_ALL_CALLS
