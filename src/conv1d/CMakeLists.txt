function(add_executable_script EXE_NAME)
  add_executable(${EXE_NAME} ${ARGN})
  set_target_properties(${EXE_NAME} PROPERTIES
      CXX_STANDARD 20
      CXX_EXTENSIONS OFF
  )

  if (APPLE)
    # Use Accelerate
    target_compile_definitions(${EXE_NAME} PRIVATE -DACCELERATE_NEW_LAPACK)
    target_link_libraries(${EXE_NAME} PRIVATE "-framework Accelerate")
  else()
    # Use OpenBLAS
    target_link_libraries(${EXE_NAME} PRIVATE OpenBLAS::OpenBLAS)
  endif()

  target_link_libraries(${EXE_NAME} PRIVATE
    fmt::fmt
    armadillo
    Eigen3::Eigen
    FFTW3::fftw3
    FFTW3::fftw3f
    opencv_world
  )

  if (IPP_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE ${IPP_LIBRARIES})
    target_compile_definitions(${EXE_NAME} PRIVATE -DHAS_IPP)
  endif()

endfunction()


add_executable_script(conv1d main.cpp)

add_executable_script(conv1d_bench benchmarks.cpp)
target_link_libraries(conv1d_bench PRIVATE benchmark::benchmark benchmark::benchmark_main)
