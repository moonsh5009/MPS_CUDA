file(GLOB CPP_SOURCES "*.cpp")
file(GLOB CU_SOURCES "*.cu")
file(GLOB HEADERS "*.h" "*.cuh")

mps_add_shared_library(
    NAME MCore_simulate
    BUILD_DEFINE BUILD_MCORE_SIMULATE
    SOURCES ${CPP_SOURCES}
    CUDA_SOURCES ${CU_SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES MCore_interface MCore_util
    PCH_HEADER stdafx.h
    CUDA_SEPARABLE
)
