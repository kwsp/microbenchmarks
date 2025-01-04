set(TARGET matrix_conversion_benchmark)

add_executable(${TARGET} matrix_conversion_benchmark.cpp)
set_target_properties(${TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

target_link_libraries(${TARGET} 
    armadillo 
    benchmark::benchmark
    opencv_world
)

if(OpenMP_CXX_FOUND) 
    target_compile_definitions(${TARGET} PRIVATE HAS_OPENMP)
    target_link_libraries(${TARGET} OpenMP::OpenMP_CXX)
endif()