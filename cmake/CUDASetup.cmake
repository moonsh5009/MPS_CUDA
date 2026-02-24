set(CMAKE_CUDA_ARCHITECTURES "89")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Suppress nvcc warnings
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1388>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1394>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=3189>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20013>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015>
)
