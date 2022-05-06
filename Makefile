
use_omp = 1
use_simd = 1

ifeq (use_omp, 1)
	_omp = -DKokkos_ENABLE_OPENMP=ON
else
	_omp = -DKokkos_ENABLE_OPENMP=OFF
endif

ifeq (use_simd, 1)
	_simd = -DKokkos_ENABLE_SIMD=ON
else
	_simd = -DKokkos_ENABLE_SIMD=OFF
endif

RUN_DIR = ./data
args =

.PHONY: setup-cuda run-cuda setup-hip run-hip setup-omp run-omp setup-serial run-serial FORCE

FORCE:

./cmake-build-cuda/src/armon_cuda.exe: FORCE
	cd ./cmake-build-cuda && make -j4 && cd ..

./cmake-build-hip/src/armon_hip.exe: FORCE
	cd ./cmake-build-cuda && make -j4 && cd ..

./cmake-build-openmp/src/armon_openmp.exe: FORCE
	cd ./cmake-build-openmp && make -j4 && cd ..

./cmake-build-serial/src/armon_serial.exe: FORCE
	cd ./cmake-build-serial && make -j4 && cd ..

setup-cuda:
	mkdir -p ./cmake-build-cuda
	CMAKE_BUILD_TYPE=RELEASE cd ./cmake-build-cuda && cmake .. -DKokkos_ENABLE_CUDA=ON $(_omp) $(_simd)

run-cuda: ./cmake-build-cuda/src/armon_cuda.exe
	cd $(RUN_DIR) && ../cmake-build-cuda/src/armon_cuda $(args)

setup-hip:
	mkdir -p ./cmake-build-hip
	CMAKE_BUILD_TYPE=RELEASE cd ./cmake-build-hip  && cmake .. -DKokkos_ENABLE_HIP=ON $(_omp) $(_simd)

run-hip: ./cmake-build-hip/src/armon_hip.exe
	cd $(RUN_DIR) && ../cmake-build-cuda/src/armon_cuda $(args)

setup-omp:
	mkdir -p ./cmake-build-openmp
	CMAKE_BUILD_TYPE=RELEASE cd ./cmake-build-openmp && cmake .. -DKokkos_ENABLE_OPENMP=ON $(_simd)

run-omp: ./cmake-build-openmp/src/armon_openmp.exe
	cd $(RUN_DIR) && ../cmake-build-openmp/src/armon_openmp $(args)

setup-serial:
	mkdir -p ./cmake-build-serial
	CMAKE_BUILD_TYPE=RELEASE cd ./cmake-build-serial && cmake .. && make -j4

run-serial: ./cmake-build-serial/src/armon_serial.exe
	cd $(RUN_DIR) && ../cmake-build-serial/src/armon_serial $(args)
