file(GLOB CPP_SOURCES "*.cpp")
file(GLOB CU_SOURCES "*.cu")
file(GLOB HEADERS "*.h" "*.cuh")

mps_add_shared_library(
    NAME MCore_util
    BUILD_DEFINE BUILD_MCORE_UTIL
    SOURCES ${CPP_SOURCES}
    CUDA_SOURCES ${CU_SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES Vulkan::Vulkan CUDA::cudart CUDA::cuda_driver glm eigen glfw3
    PCH_HEADER stdafx.h
    CUDA_SEPARABLE
)
