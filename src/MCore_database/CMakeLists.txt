file(GLOB CPP_SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

mps_add_shared_library(
    NAME MCore_database
    BUILD_DEFINE BUILD_MCORE_DATABASE
    SOURCES ${CPP_SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES MCore_interface
    PCH_HEADER stdafx.h
)
