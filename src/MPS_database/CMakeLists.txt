file(GLOB CPP_SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

mps_add_shared_library(
    NAME MPS_database
    BUILD_DEFINE BUILD_MPS_DATABASE
    SOURCES ${CPP_SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES MCore_database
    PCH_HEADER stdafx.h
)
