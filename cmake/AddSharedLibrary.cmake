function(mps_add_shared_library)
    cmake_parse_arguments(LIB
        "CUDA_SEPARABLE"
        "NAME;BUILD_DEFINE;PCH_HEADER"
        "SOURCES;CUDA_SOURCES;HEADERS;DEPENDENCIES"
        ${ARGN}
    )

    add_library(${LIB_NAME} SHARED ${LIB_SOURCES} ${LIB_CUDA_SOURCES} ${LIB_HEADERS})

    target_compile_definitions(${LIB_NAME} PRIVATE ${LIB_BUILD_DEFINE})

    target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

    if(LIB_DEPENDENCIES)
        target_link_libraries(${LIB_NAME} PUBLIC ${LIB_DEPENDENCIES})
    endif()

    if(LIB_CUDA_SEPARABLE AND LIB_CUDA_SOURCES)
        set_target_properties(${LIB_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()

    if(LIB_PCH_HEADER)
        target_precompile_headers(${LIB_NAME} PRIVATE ${LIB_PCH_HEADER})
    endif()
endfunction()
