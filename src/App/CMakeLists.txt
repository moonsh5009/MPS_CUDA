file(GLOB CPP_SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_executable(App WIN32 ${CPP_SOURCES} ${HEADERS})

target_include_directories(App PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(App PRIVATE
    MCore_util
    MCore_system
    MPS_database
    MPS_simulate
    MPS_system
)

target_precompile_headers(App PRIVATE stdafx.h)

# Copy shader files to output directory
add_custom_target(CopyShaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/shader/header
        $<TARGET_FILE_DIR:App>/shader/header
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/shader/render
        $<TARGET_FILE_DIR:App>/shader/render
    COMMENT "Copying shader files..."
)

# Copy shader_builder to output directory
add_custom_target(CopyShaderBuilder ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/MCore_render/shader_builder
        $<TARGET_FILE_DIR:App>/shader_builder
    COMMENT "Copying shader builder..."
)

add_dependencies(App CopyShaders CopyShaderBuilder)
