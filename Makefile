
use_omp ?= 1
use_simd ?= 1
use_single ?= 0

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

build_type ?= RELEASE
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
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-cuda && cmake .. -DKokkos_ENABLE_CUDA=ON $(_omp) $(_simd) $(_flt) && $(MAKE) $(make_args) clean

run-cuda: ./cmake-build-cuda/src/armon_cuda.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-cuda/src/armon_cuda.exe $(args_)

build-hip:
	@mkdir -p ./cmake-build-hip
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-hip  && cmake .. -DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON $(_omp) $(_simd) $(_flt) && $(MAKE) $(make_args) clean

run-hip: ./cmake-build-hip/src/armon_hip.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-hip/src/armon_hip.exe $(args_)

build-omp:
	@mkdir -p ./cmake-build-openmp
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-openmp && cmake .. -DKokkos_ENABLE_OPENMP=ON $(_simd) $(_flt) && $(MAKE) $(make_args) clean

run-omp: ./cmake-build-openmp/src/armon_openmp.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-openmp/src/armon_openmp.exe $(args_)

build-serial:
	@mkdir -p ./cmake-build-serial
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-serial && cmake .. $(_flt) && $(MAKE) $(make_args) clean

run-serial: ./cmake-build-serial/src/armon_serial.exe $(RUN_DIR)
	cd $(RUN_DIR) && ../cmake-build-serial/src/armon_serial.exe $(args_)
