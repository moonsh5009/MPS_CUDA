file(GLOB CPP_SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

mps_add_shared_library(
    NAME MCore_system
    BUILD_DEFINE BUILD_MCORE_SYSTEM
    SOURCES ${CPP_SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES MCore_render MCore_simulate MCore_database MCore_interface
    PCH_HEADER stdafx.h
)
