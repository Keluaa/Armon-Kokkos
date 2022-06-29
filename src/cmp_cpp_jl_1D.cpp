
#include "armon.cpp"


Params* p = nullptr;
Data* d = nullptr;

flt_t** d_ptr = nullptr;

bool kokkos_initialized = false;


extern "C" void init_cmp(int test, int scheme, int nb_ghost, int nb_cells, flt_t cfl, flt_t Dt,
                         bool cst_dt, bool euler_projection)
{
    if (p != nullptr) {
        delete p;
        delete d;
        delete[] d_ptr;
    }

    int argc = 0;

    if (!kokkos_initialized) {
        Kokkos::initialize(argc, nullptr);
        kokkos_initialized = true;
    }

    printf("Using Armon-Kokkos compiled the " __DATE__ " at " __TIME__ "\n");
    fflush(stdout);

    p = new Params();

    switch (test) {
    case 0: p->test = Params::Test::Sod;       break;
    case 1: p->test = Params::Test::Bizarrium; break;
    default: printf("Wrong test: %d\n", test); exit(1);
    }

    switch (scheme) {
    case 0: p->scheme = Params::Scheme::Godunov;    break;
    case 1: p->scheme = Params::Scheme::GAD_minmod; break;
    default: printf("Wrong scheme: %d\n", scheme); exit(1);
    }

    p->riemann = Params::Riemann::Acoustic;

    p->nb_ghosts = nb_ghost;
    p->nb_cells = nb_cells;
    p->cfl = cfl;
    p->Dt = Dt;
    p->cst_dt = cst_dt;
    p->euler_projection = euler_projection;

    d = new Data("data", p->nb_cells + 2 * p->nb_ghosts);

    d_ptr = new flt_t*[16]{
        d->x.data(), d->X.data(),
        d->rho.data(), d->umat.data(), d->emat.data(), d->Emat.data(), d->pmat.data(), d->cmat.data(), d->gmat.data(),
        d->ustar.data(), d->pstar.data(), d->ustar_1.data(), d->pstar_1.data(),
        d->tmp_rho.data(), d->tmp_urho.data(), d->tmp_Erho.data()
    };
}


extern "C" void init_test()
{
    init_test(*p, *d);

//    for (int i = 0; i < p->nb_cells + 2 * p->nb_ghosts; i++) {
//        d->tmp_rho[i] = 0;
//        d->tmp_Erho[i] = 0;
//        d->tmp_urho[i] = 0;
//    }
}


extern "C" void end_cmp()
{
//    Kokkos::finalize();
}


extern "C" flt_t** get_data()
{
    return d_ptr;
}


int cycles = 0;
flt_t t = 0., dta = 0., dt = 0.;


extern "C" void step_init_loop()
{
    cycles = 0;
    t = 0.;
    dta = 0.;
    dt = 0.;
}


extern "C" flt_t step_dtCFL()
{
    dt = dtCFL(*p, *d, dta);
    return dt;
}


extern "C" void step_boundary_conditions()
{
    boundaryConditions(*p, *d);
}


extern "C" void step_numerical_fluxes()
{
    numericalFluxes(*p, *d, dt);
}


extern "C" void step_cell_update()
{
    cellUpdate(*p, *d, dt);
}


extern "C" void step_euler_projection()
{
    first_order_euler_remap(*p, *d, dt);
}


extern "C" void step_update_EOS()
{
    update_EOS(*p, *d);
}


extern "C" void step_end_loop()
{
    dta = dt;
    cycles++;
    t += dt;
}
