if(MSVC)
    # C/C++ only flags (exclude CUDA which uses nvcc)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:/W3>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4251>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4275>"
    )
    add_compile_definitions(_UNICODE UNICODE)

    # Per-config flags (each flag as separate expression for VS generator)
    add_compile_options(
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Debug>>:/Od>"
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Debug>>:/RTC1>"
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Debug>>:/Zi>"
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:/O2>"
        "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:/Oi>"
    )
endif()
