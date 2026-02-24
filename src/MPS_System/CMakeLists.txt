file(GLOB CPP_SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

mps_add_shared_library(
    NAME MPS_system
    BUILD_DEFINE BUILD_MPS_SYSTEM
    SOURCES ${CPP_SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES MCore_system MPS_simulate
    PCH_HEADER stdafx.h
)
