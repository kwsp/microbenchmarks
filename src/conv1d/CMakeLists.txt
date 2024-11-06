find_package(fmt CONFIG REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)
find_path(FFTCONV_INCLUDE_DIRS "fftconv.hpp")

function(add_executable_script EXE_NAME)
  add_executable(${EXE_NAME} ${ARGN})
  set_target_properties(${EXE_NAME} PROPERTIES
      CXX_STANDARD 20
      CXX_EXTENSIONS OFF
  )

  target_link_libraries(${EXE_NAME} PRIVATE
    fmt::fmt
    armadillo
    FFTW3::fftw3
    FFTW3::fftw3f
  )
  target_include_directories(${EXE_NAME} PRIVATE ${FFTCONV_INCLUDE_DIRS})

  if (APPLE)
    # Use Accelerate
    target_compile_definitions(${EXE_NAME} PRIVATE -DACCELERATE_NEW_LAPACK)
    target_link_libraries(${EXE_NAME} PRIVATE "-framework Accelerate")
  else()
    # Use OpenBLAS
    target_link_libraries(${EXE_NAME} PRIVATE OpenBLAS::OpenBLAS)
  endif()

endfunction()


add_executable_script(conv1d main.cpp)

add_executable_script(conv1d_benchmarks benchmarks.cpp)
target_link_libraries(conv1d_benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main)