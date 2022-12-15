
using Printf
using Statistics


mutable struct KokkosOptions
    scheme::String
    riemann::String
    riemann_limiter::String
    nghost::Int
    cfl::Float64
    Dt::Float64
    maxtime::Float64
    maxcycle::Int
    projection::String
    cst_dt::Bool
    ieee_bits::Int
    silent::Int
    output_file::String
    write_output::Bool
    write_ghosts::Bool
    use_simd::Bool
    gpu::String
    num_threads::Int
    threads_places::String
    threads_proc_bind::String
    dimension::Int
    axis_splitting::Vector{String}
    tests::Vector{String}
    cells_list::Vector{Vector{Int}}
    base_file_name::String
    gnuplot_script::String
    repeats::Int
    compiler::String
end


project_dir = @__DIR__()
run_dir = project_dir * "/../data"
make_options = ["--no-print-directory"]

# Julia adds its libs to the ENV, which can interfere with cmake
cmake_env = copy(ENV)
delete!(cmake_env, "LD_LIBRARY_PATH")


function KokkosOptions(;
        scheme = "GAD", riemann = "acoustic", riemann_limiter = "minmod",
        nghost = 2, cfl = 0.6, Dt = 0., maxtime = 0.0,
        maxcycle = 500, projection = "euler", cst_dt = false, ieee_bits = 64,
        silent = 2, output_file = "output", write_output = false, write_ghosts = false,
        use_simd = true, gpu = "", 
        num_threads = 1, threads_places = "cores", threads_proc_bind = "close",
        dimension = 2, axis_splitting = [], tests = [], cells_list = [],
        base_file_name = "", gnuplot_script = "", repeats = 1,
        compiler = "clang")
    return KokkosOptions(
        scheme, riemann, riemann_limiter, 
        nghost, cfl, Dt, maxtime, maxcycle, 
        projection, cst_dt, ieee_bits, 
        silent, output_file, write_output, write_ghosts, 
        use_simd, gpu, 
        num_threads, threads_places, threads_proc_bind, 
        dimension, axis_splitting, tests, cells_list, 
        base_file_name, gnuplot_script, repeats,
        compiler
    )
end


function parse_arguments(args::Vector{String})
    options = KokkosOptions()

    dim_index = findfirst(x -> x == "--dim", args)
    if !isnothing(dim_index)
        options.dimension = parse(Int, args[dim_index+1])
        if options.dimension ∉ (1, 2)
            error("Unexpected dimension: $(options.dimension)")
        end
        deleteat!(args, [dim_index, dim_index+1])
    end

    i = 1
    while i <= length(args)
        arg = args[i]
    
        # Solver params
        if arg == "-s"
            options.scheme = replace(args[i+1], '-' => '_')
            i += 1
        elseif arg == "--ieee"
            options.ieee_bits = parse(Int, args[i+1])
            i += 1
        elseif arg == "--cycle"
            options.maxcycle = parse(Int, args[i+1])
            i += 1
        elseif arg == "--riemann"
            options.riemann = replace(args[i+1], '-' => '_')
            i += 1
        elseif arg == "--riemann-limiter"
            options.riemann_limiter = replace(args[i+1], '-' => '_')
            i += 1
        elseif arg == "--time"
             options.maxtime = parse(Float64, args[i+1])
            i += 1
        elseif arg == "--cfl"
            options.cfl = parse(Float64, args[i+1])
            i += 1
        elseif arg == "--dt"
            options.Dt = parse(Float64, args[i+1])
            i += 1
        elseif arg == "--projection"
            options.projection = args[i+1]
            i += 1
        elseif arg == "--cst-dt"
            options.cst_dt = parse(Bool, args[i+1])
            i += 1
        elseif arg == "--nghost"
            options.nghost = parse(Int, args[i+1])
            i += 1
    
        # Solver output params
        elseif arg == "--verbose"
            options.silent = parse(Int, args[i+1])
            i += 1
        elseif arg == "--output-file"
            options.output_file = args[i+1]
            i += 1
        elseif arg == "--write-output"
            options.write_output = parse(Bool, args[i+1])
            i += 1
        elseif arg == "--write-ghosts"
            options.write_ghosts = parse(Bool, args[i+1])
            i += 1
    
        # Multithreading params
        elseif arg == "--use-simd"
            options.use_simd = parse(Bool, args[i+1])
            i += 1
        elseif arg == "--num-threads"
            options.num_threads = parse(Int, args[i+1])
            i += 1
        elseif arg == "--threads-places"
            options.threads_places = args[i+1]
            i += 1
        elseif arg == "--threads-proc-bind"
            options.threads_proc_bind = args[i+1]
            i += 1
    
        # GPU params
        elseif arg == "--gpu"
            options.gpu = uppercase(args[i+1])
            i += 1
    
        # 2D only params
        elseif arg == "--splitting"
            if options.dimension != 2
                error("'--splitting' is 2D only")
            end
            options.axis_splitting = split(args[i+1], ',')
            i += 1
    
        # List params
        elseif arg == "--tests"
            options.tests = split(args[i+1], ',')
            i += 1
        elseif arg == "--cells-list"
            if options.dimension == 1
                list = split(args[i+1], ',')
                options.cells_list = convert.(Int, parse.(Float64, list))
            else
                domains_list = split(args[i+1], ';')
                domains_list = split.(domains_list, ',')
                domains_list_t = Vector{Vector{Int}}(undef, length(domains_list))
                for (i, domain) in enumerate(domains_list)
                    domains_list_t[i] = convert.(Int, parse.(Float64, domain))
                end
                options.cells_list = domains_list_t
            end
            i += 1
        
        # MPI params
        elseif arg == "--single-comm"
            options.single_comm_per_axis_pass = parse(Bool, args[i+1])
            i += 1
    
        # Measurements params
        elseif arg == "--repeats"
            options.repeats = parse(Int, args[i+1])
            i += 1
    
        # Measurement output params
        elseif arg == "--data-file"
            options.base_file_name = args[i+1]
            i += 1
        elseif arg == "--gnuplot-script"
            options.gnuplot_script = args[i+1]
            i += 1

        # Kokkos backend options
        elseif arg == "--compiler"
            options.compiler = args[i+1]
            i += 1

        else
            error("Wrong option: ", arg)
        end

        i += 1
    end

    if options.dimension == 1
        options.axis_splitting = ["Sequential"]
    end

    return options
end


function run_cmd_print_on_error(cmd::Cmd)
    # Run a command without printing its output to stdout
    # However if there is an error we print the command's output
    # A temporary file is used to achieve that
    mktemp() do _, file
        try 
            run(pipeline(cmd, stdout=file#= , stderr=file =#))
        catch
            flush(file)
            seekstart(file)
            println("ERROR:\n", read(file, String))
            rethrow()
        end
    end
end


function init_cmake(options::KokkosOptions)
    use_SIMD = options.use_simd
    use_single_presicion = options.ieee_bits == 32
    compiler = lowercase(options.compiler)
    dim = options.dimension

    cmake_options = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DKokkos_ENABLE_OPENMP=ON",
        "-DKokkos_ENABLE_SIMD=$(use_SIMD ? "ON" : "OFF")",
        "-DUSE_SINGLE_PRECISION=$(use_single_presicion ? "ON" : "OFF")",
    ]

    if options.gpu == "CUDA"
        build_dir = project_dir * "/cmake-build-cuda"
        target_exe = "armon_cuda"
        push!(cmake_options, "-DKokkos_ENABLE_CUDA=ON")
    elseif options.gpu == "ROCM"
        build_dir = project_dir * "/cmake-build-hip"
        target_exe = "armon_hip"
        append!(cmake_options, [
            "-DKokkos_ENABLE_HIP=ON",
            # Override the C++ compiler for the one required by HIP
            "-DCMAKE_C_COMPILER=hipcc",
            "-DCMAKE_CXX_COMPILER=hipcc"
        ])
    else
        build_dir = project_dir * "/cmake-build-openmp"
        target_exe = "armon_openmp"
    end

    if options.gpu != "ROCM"
        if compiler == "icc"
            c_compiler = "icc"
            cpp_compiler = "icpc"
        elseif compiler == "gcc"
            c_compiler = "gcc"
            cpp_compiler = "g++"
        elseif compiler == "clang"
            c_compiler = "clang"
            cpp_compiler = "clang++"
        else
            error("Unknown compiler: $compiler")
        end

        append!(cmake_options, [
            "-DCMAKE_C_COMPILER=$c_compiler",
            "-DCMAKE_CXX_COMPILER=$cpp_compiler",
        ])
    end

    if dim == 2
        target_exe *= "_2D"
    end

    target_exe *= ".exe"

    mkpath(build_dir)
    rm(build_dir * "/CMakeCache.txt"; force=true)

    run_cmd_print_on_error(Cmd(`cmake $cmake_options ..`; env=cmake_env, dir=build_dir))
    run_cmd_print_on_error(Cmd(`make $make_options clean`; env=cmake_env, dir=build_dir))
    
    return build_dir, target_exe
end


function compile_backend(build_dir, target_exe)
    run_cmd_print_on_error(Cmd(`make $make_options $target_exe`; env=cmake_env, dir=build_dir))
    return build_dir * "/src/$target_exe"
end


function setup_env(options::KokkosOptions)
    ENV["OMP_PLACES"] = options.threads_places
    ENV["OMP_PROC_BIND"] = options.threads_proc_bind
    ENV["OMP_NUM_THREADS"] = options.num_threads
end


function build_args_list(options::KokkosOptions)
    limiter = lowercase(options.riemann_limiter)
    if limiter == "no_limiter"
        limiter = "None"
    elseif limiter == "minmod"
        limiter = "Minmod"
    elseif limiter == "superbee"
        limiter = "Superbee"
    else
        error("Wrong limiter: $limiter")
    end

    return [
        "-s", options.scheme,
        "--riemann", options.riemann,
        "--limiter", limiter,
        "--nghost", options.nghost,
        "--cfl", options.cfl,
        "--dt", options.Dt,
        "--time", options.maxtime,
        "--cycle", options.maxcycle,
        "--projection", options.projection,
        "--cst-dt", options.cst_dt,
        "--verbose", options.silent,
        "--output", options.output_file,
        "--write-output", Int(options.write_output),
        "--write-ghosts", Int(options.write_ghosts),
        "--single-comm", "0"
    ]
end


function get_run_command(exe_path, args)
    return Cmd(`$exe_path $args`; dir=run_dir)
end


function run_and_parse_output(cmd::Cmd, verbose::Bool, repeats::Int)
    if verbose
        println(cmd)
    end

    total_giga_cells_per_sec = 0

    for _ in 1:repeats
        output = read(cmd, String)

        mega_cells_per_sec_raw = match(r"Cells/sec:\s*\K[0-9\.]+", output)

        if isnothing(mega_cells_per_sec_raw)
            error("Command failed, wrong output: ", cmd, "\nOutput:\n", output)
        elseif verbose
            println(output)
        end

        mega_cells_per_sec = parse(Float64, mega_cells_per_sec_raw.match)
        giga_cells_per_sec = mega_cells_per_sec / 1e3
        total_giga_cells_per_sec += giga_cells_per_sec
    end

    total_giga_cells_per_sec /= repeats

    return total_giga_cells_per_sec
end


function run_armon(options::KokkosOptions, verbose::Bool)
    build_dir, target_exe = init_cmake(options)
    exe_path = compile_backend(build_dir, target_exe)
    setup_env(options)
    base_args = build_args_list(options)

    for test in options.tests, axis_splitting in options.axis_splitting
        if isempty(options.base_file_name)
            data_file_name = ""
        elseif options.dimension == 1
            data_file_name = options.base_file_name * test * ".csv"
        else
            data_file_name = options.base_file_name * test

            if length(options.axis_splitting) > 1
                data_file_name *= "_" * axis_splitting
            end
            
            data_file_name *= ".csv"
        end

        for cells in options.cells_list
            args = Any[
                "-t", test,
                "--splitting", axis_splitting,
                "--cells", join(cells, ',')
            ]
            append!(args, base_args)
    
            if options.dimension == 1
                @printf(" - %s, %10g cells: ", test, cells[1])
            else
                @printf(" - %s, %-14s %10g cells (%5gx%-5g): ",
                    test, axis_splitting, prod(cells), cells[1], cells[2])
            end
    
            run_cmd = get_run_command(exe_path, args)
            cells_throughput = run_and_parse_output(run_cmd, verbose, options.repeats)
            
            @printf("%.2f Giga cells/sec\n", cells_throughput)
            
            if !isempty(data_file_name)
                # Append the result to the output file
                open(data_file_name, "a") do file
                    println(file, prod(cells), ", ", cells_throughput)
                end
            end
            
            if !isempty(options.gnuplot_script)
                # Update the plot
                # We redirect the output of gnuplot to null so that there is no warning messages displayed
                run(pipeline(`gnuplot $(options.gnuplot_script)`, stdout=devnull, stderr=devnull))
            end
        end
    end
end


if !isinteractive()
    kokkos_options = parse_arguments(ARGS)
    run_armon(kokkos_options, false)
end
