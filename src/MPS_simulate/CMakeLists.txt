file(GLOB CPP_SOURCES "*.cpp")
file(GLOB CU_SOURCES "*.cu")
file(GLOB HEADERS "*.h" "*.cuh")

mps_add_shared_library(
    NAME MPS_simulate
    BUILD_DEFINE BUILD_MPS_SIMULATE
    SOURCES ${CPP_SOURCES}
    CUDA_SOURCES ${CU_SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES MCore_simulate MCore_util
    PCH_HEADER stdafx.h
    CUDA_SEPARABLE
)
