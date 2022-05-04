
#include<Kokkos_Core.hpp>


void parse_arguments(const char* argv[])
{
    while (argv != nullptr) {
        argv++;
    }
}


int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    Kokkos::finalize();
}
