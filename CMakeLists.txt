cmake_minimum_required(VERSION 3.28)
project(PhysicsSimulation LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 12.6 REQUIRED)
find_package(Vulkan REQUIRED)

include(cmake/CUDASetup.cmake)
include(cmake/CompilerSettings.cmake)
include(cmake/OutputDirectories.cmake)
include(cmake/AddSharedLibrary.cmake)

# Bundled dependencies - GLM (header-only)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Bundled dependencies - Eigen (header-only)
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Bundled dependencies - GLFW (pre-built shared)
add_library(glfw3 IMPORTED SHARED GLOBAL)
set_target_properties(glfw3 PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/glfw/glfw3.dll
    IMPORTED_IMPLIB ${CMAKE_SOURCE_DIR}/lib/glfw/glfw3dll.lib
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(src)

# Compile GLSL shaders to SPIR-V
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_custom_target(CompileShaders ALL
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/src/MCore_render/shader_builder/compile.py
            ${CMAKE_SOURCE_DIR}/src/shader/
            ${CMAKE_SOURCE_DIR}/bin/x64/$<CONFIG>/shader/
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src/MCore_render/shader_builder
        COMMENT "Compiling GLSL shaders to SPIR-V..."
    )
endif()
