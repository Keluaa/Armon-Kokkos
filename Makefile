
use_omp ?= 1
use_simd ?= 1
use_single ?= 0
compiler ?=

ifeq ($(use_omp), 1)
	_omp = -DKokkos_ENABLE_OPENMP=ON
else
	_omp = -DKokkos_ENABLE_OPENMP=OFF
endif

ifeq ($(use_simd), 1)
	_simd = -DKokkos_ENABLE_SIMD=ON
else
	_simd = -DKokkos_ENABLE_SIMD=OFF
endif

ifeq ($(use_single), 1)
	_flt = -DUSE_SINGLE_PRECISION=ON
else
	_flt = -DUSE_SINGLE_PRECISION=OFF
endif

ifeq ($(compiler), icc)
	comp = -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc	
else ifeq ($(compiler), gcc)
	comp = -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
else ifeq ($(compiler), clang)
	comp = -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
else ifeq ($(compiler),)
$(info "Using default CMake compiler")
else 
$(error "Wrong compiler: " $(compiler))
endif 

build_type ?= Release
make_args ?= --no-print-directory

RUN_DIR ?= ./data
args ?=

# Remove any potential quotes around the args
args_ = $(subst $\",,$(args))

.PHONY: setup-cuda run-cuda setup-hip run-hip setup-omp run-omp setup-serial run-serial FORCE

FORCE:

$(RUN_DIR):
	mkdir -p $(RUN_DIR)

./cmake-build-cuda/src/armon_cuda.exe: FORCE
	@cd ./cmake-build-cuda && $(MAKE) $(make_args) && cd ..

./cmake-build-hip/src/armon_hip.exe: FORCE
	@cd ./cmake-build-hip && $(MAKE) $(make_args) && cd ..

./cmake-build-openmp/src/armon_openmp.exe: FORCE
	@cd ./cmake-build-openmp && $(MAKE) $(make_args) && cd ..

./cmake-build-serial/src/armon_serial.exe: FORCE
	@cd ./cmake-build-serial && $(MAKE) $(make_args) && cd ..

build-cuda:
	@mkdir -p ./cmake-build-cuda
	rm -f ./cmake-build-cuda/CMakeCache.txt
	cd ./cmake-build-cuda && cmake -DCMAKE_BUILD_TYPE=$(build_type) -DKokkos_ENABLE_CUDA=ON $(comp) $(_omp) $(_simd) $(_flt) .. && $(MAKE) $(make_args) clean

run-cuda: ./cmake-build-cuda/src/armon_cuda.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-cuda/src/armon_cuda.exe $(args_)

build-hip:
	@mkdir -p ./cmake-build-hip
	rm -f ./cmake-build-hip/CMakeCache.txt
	cd ./cmake-build-hip  && cmake -DCMAKE_BUILD_TYPE=$(build_type) -DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON $(_omp) $(_simd) $(_flt) .. && $(MAKE) $(make_args) clean

run-hip: ./cmake-build-hip/src/armon_hip.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-hip/src/armon_hip.exe $(args_)

build-omp:
	@mkdir -p ./cmake-build-openmp
	rm -f ./cmake-build-openmp/CMakeCache.txt
	cd ./cmake-build-openmp && cmake -DCMAKE_BUILD_TYPE=$(build_type) -DKokkos_ENABLE_OPENMP=ON $(comp) $(_simd) $(_flt) .. && $(MAKE) $(make_args) clean

run-omp: ./cmake-build-openmp/src/armon_openmp.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-openmp/src/armon_openmp.exe $(args_)

build-serial:
	@mkdir -p ./cmake-build-serial
	rm -f ./cmake-build-serial/CMakeCache.txt
	cd ./cmake-build-serial && cmake -DCMAKE_BUILD_TYPE=$(build_type) $(comp) $(_flt) .. && $(MAKE) $(make_args) clean

run-serial: ./cmake-build-serial/src/armon_serial.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-serial/src/armon_serial.exe $(args_)

