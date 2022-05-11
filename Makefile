
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
make_args ?= -j4 --no-print-directory

RUN_DIR ?= ./data
args ?=

.PHONY: setup-cuda run-cuda setup-hip run-hip setup-omp run-omp setup-serial run-serial FORCE

FORCE:

./cmake-build-cuda/src/armon_cuda.exe: FORCE
	@cd ./cmake-build-cuda && make $(make_args) && cd ..

./cmake-build-hip/src/armon_hip.exe: FORCE
	@cd ./cmake-build-hip && make $(make_args) && cd ..

./cmake-build-openmp/src/armon_openmp.exe: FORCE
	@cd ./cmake-build-openmp && make $(make_args) && cd ..

./cmake-build-serial/src/armon_serial.exe: FORCE
	@cd ./cmake-build-serial && make $(make_args) && cd ..

build-cuda:
	@mkdir -p ./cmake-build-cuda
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-cuda && cmake .. -DKokkos_ENABLE_CUDA=ON $(_omp) $(_simd) $(_flt) && make clean $(make_args)

run-cuda: ./cmake-build-cuda/src/armon_cuda.exe
	cd $(RUN_DIR) && ../cmake-build-cuda/src/armon_cuda.exe $(args)

build-hip:
	@mkdir -p ./cmake-build-hip
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-hip  && cmake .. -DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON $(_omp) $(_simd) $(_flt) && make clean $(make_args)

run-hip: ./cmake-build-hip/src/armon_hip.exe
	cd $(RUN_DIR) && ../cmake-build-hip/src/armon_hip.exe $(args)

build-omp:
	@mkdir -p ./cmake-build-openmp
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-openmp && cmake .. -DKokkos_ENABLE_OPENMP=ON $(_simd) $(_flt) && make clean $(make_args)

run-omp: ./cmake-build-openmp/src/armon_openmp.exe
	cd $(RUN_DIR) && ../cmake-build-openmp/src/armon_openmp.exe $(args)

build-serial:
	@mkdir -p ./cmake-build-serial
	CMAKE_BUILD_TYPE=$(build_type) cd ./cmake-build-serial && cmake .. $(_flt) && make clean $(make_args)

run-serial: ./cmake-build-serial/src/armon_serial.exe
	cd $(RUN_DIR) && ../cmake-build-serial/src/armon_serial.exe $(args)
