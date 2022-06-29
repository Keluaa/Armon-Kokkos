
#include "armon_2D.cpp"


Params* p = nullptr;
Data* d = nullptr;

flt_t** d_ptr = nullptr;


bool kokkos_initialized = false;

extern "C" void init_cmp(int test, int scheme, int nb_ghost, int nx, int ny, flt_t cfl, flt_t Dt,
                         bool cst_dt, bool euler_projection, bool transpose_dims, int axis_splitting)
{
    delete p;
    delete d;
    delete[] d_ptr;

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
    case 1: p->test = Params::Test::Sod_y;     break;
    case 2: p->test = Params::Test::Sod_circ;  break;
    case 3: p->test = Params::Test::Bizarrium; break;
    default: printf("Wrong test: %d\n", test); exit(1);
    }

    switch (scheme) {
    case 0: p->scheme = Params::Scheme::Godunov;    break;
    case 1: p->scheme = Params::Scheme::GAD_minmod; break;
    default: printf("Wrong scheme: %d\n", scheme); exit(1);
    }

    p->riemann = Params::Riemann::Acoustic;

    p->nb_ghosts = nb_ghost;
    p->nx = nx;
    p->ny = ny;
    p->cfl = cfl;
    p->Dt = Dt;
    p->cst_dt = cst_dt;
    p->euler_projection = euler_projection;
    p->transpose_dims = transpose_dims;

    switch (axis_splitting) {
    case 0: p->axis_splitting = Params::AxisSplitting::Sequential;    break;
    case 1: p->axis_splitting = Params::AxisSplitting::SequentialSym; break;
    case 2: p->axis_splitting = Params::AxisSplitting::Strang;        break;
    default: printf("Wrong splitting: %d\n", axis_splitting); exit(1);
    }

    p->init_indexing();

    d = new Data("data", p->nb_cells);

    d_ptr = new flt_t*[19]{
        d->x.data(), d->y.data(),
        d->rho.data(), d->umat.data(), d->vmat.data(), d->Emat.data(), d->pmat.data(), d->cmat.data(), d->gmat.data(),
        d->ustar.data(), d->pstar.data(), d->ustar_1.data(), d->pstar_1.data(),
        d->tmp_rho.data(), d->tmp_urho.data(), d->tmp_vrho.data(), d->tmp_Erho.data(),
        d->domain_mask.data(), d->domain_mask_T.data()
    };
}


extern "C" void init_test()
{
    init_test(*p, *d);

    for (int i = 0; i < p->nb_cells; i++) {
        d->tmp_rho[i] = 0;
        d->tmp_Erho[i] = 0;
        d->tmp_urho[i] = 0;
        d->tmp_vrho[i] = 0;
    }
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
int last_i = 0;
view *x, *u, *mask;

std::vector<std::pair<Params::Axis, flt_t>> axis_splitting_vec;
std::vector<std::pair<Params::Axis, flt_t>>::const_iterator axis_splitting_iter;
Params::Axis axis;
flt_t dt_factor;


extern "C" void step_init_loop()
{
    auto&& [last_i_val, x_ref, u_ref, mask_ref] = update_axis_parameters(*p, *d, p->current_axis);
    last_i = last_i_val;
    x = &x_ref;
    u = &u_ref;
    mask = &mask_ref;

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


extern "C" void step_init_split_axes()
{
    axis_splitting_vec = split_axes(*p, cycles);
    axis_splitting_iter = axis_splitting_vec.cbegin();
}


extern "C" void step_step_split_axes()
{
    std::tie(axis, dt_factor) = *axis_splitting_iter;
    axis_splitting_iter++;

    auto&& [last_i_val, x_ref, u_ref, mask_ref] = update_axis_parameters(*p, *d, axis);
    last_i = last_i_val;
    x = &x_ref;
    u = &u_ref;
    mask = &mask_ref;
}


extern "C" void step_boundary_conditions()
{
//    printf("@i(1,   ny) = %d\n", index_1D(*p, 0      , p->ny-1));
//    printf("@i(0,   ny) = %d\n", index_1D(*p, -1     , p->ny-1));
//    printf("@i(nx,  ny) = %d\n", index_1D(*p, p->nx-1, p->ny-1));
//    printf("@i(nx+1,ny) = %d\n", index_1D(*p, p->nx  , p->ny-1));
//
//    printf("@i(nx,   1) = %d\n", index_1D(*p, p->nx-1, 0      ));
//    printf("@i(nx,   0) = %d\n", index_1D(*p, p->nx-1, -1     ));
//    printf("@i(nx,  ny) = %d\n", index_1D(*p, p->nx-1, p->ny-1));
//    printf("@i(nx,ny+1) = %d\n", index_1D(*p, p->nx-1, p->ny  ));
//
//    printf("indexing vars: ");
//    printf("row_length = %d\n", p->row_length);
//    printf("col_length = %d\n", p->col_length);
//    printf("nb_cells = %d\n", p->nb_cells);
//    printf("ideb = %d\n", p->ideb);
//    printf("ifin = %d\n", p->ifin);
//    printf("index_start = %d\n", p->index_start);
//    printf("idx_row = %d\n", p->idx_row);
//    printf("idx_col = %d\n", p->idx_col);

    boundaryConditions(*p, *d);
}


extern "C" void step_numerical_fluxes()
{
    numericalFluxes(*p, *d, dt * dt_factor, last_i, *u);
}


extern "C" void step_cell_update()
{
    cellUpdate(*p, *d, dt * dt_factor, *u, *x, *mask);
}


extern "C" void step_euler_projection()
{
    first_order_euler_remap(*p, *d, dt);
    if (p->transpose_dims)
        transpose_parameters(*p);
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
